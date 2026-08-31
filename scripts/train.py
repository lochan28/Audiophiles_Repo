"""
train.py

The training loop. Architecture-agnostic: it never imports a model directly,
it looks one up in ARCHITECTURES. Adding the TCN is one entry in that dict
plus its model file -- nothing in this loop changes.

    python scripts/train.py --arch crn --epochs 40
    python scripts/train.py --arch crn --epochs 40 --note "wider encoder"
    python scripts/train.py --resume 0007_crn_h256_la2

Every run writes to runs/NNNN_arch_tag/ through runlog.py: settings, a copy of
the source that produced it, one metrics line per epoch, checkpoints, and
audio and mask images at evaluation points. Nothing reads back from a live
process, so a crashed run keeps everything it had logged.

Two deliberate choices about what is measured:

- TRAINING mixtures are regenerated fresh each epoch, so the model sees new
  speech-and-noise combinations constantly. VALIDATION uses the frozen
  mixtures on disk, unchanged. If validation were also random, its curve would
  wobble with the draw rather than the model and comparing epochs -- or runs --
  would be meaningless.
- The expensive perceptual metrics (PESQ, STOI) are CPU-bound and can take
  longer than a training epoch, so they run every EVAL_EVERY epochs. The cheap
  ones run every epoch.
"""

import os
import sys
import time
import json
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mixing import SpeechNoiseMixtures, FrozenMixtures, read_noise_classes
from runlog import RunLog, RUNS_DIR
import model_crn


# ═══════════════════════════════════════════════════════════════════
#  TUNING KNOBS — safe to change, this is what you experiment with
# ═══════════════════════════════════════════════════════════════════

EPOCHS = 40
# How many passes over the training set. Each pass sees fresh noise mixtures,
# so more epochs is not simply more repetition of the same examples.

BATCH_SIZE = 16
# Clips per step. A batch of 16 uses 1.62 GB of the 8 GB card, so there is
# room to raise this. LARGER is faster per epoch and gives steadier gradients;
# SMALLER adds useful noise that can help generalisation on a small dataset.
# Sensible range: 8-64.

LEARNING_RATE = 3e-4
# How big a step the optimiser takes. TOO HIGH and the loss jumps around or
# turns to NaN; TOO LOW and it crawls. 3e-4 is the usual safe starting point
# for AdamW. Sensible range: 1e-4 to 1e-3.

WEIGHT_DECAY = 1e-2
# Gently shrinks weights toward zero each step, which discourages the model
# from over-relying on any one pattern. This is a real defence against
# overfitting on 1292 clips. Sensible range: 0 to 1e-1.

OPTIMIZER = "adamw"
# One of: adamw, adam, rmsprop. PROJECT_CONTEXT_4.md Section 4 planned a
# comparison of all three; this is the knob for it.

AUX_LOSS_WEIGHT = model_crn.AUX_LOSS_WEIGHT
# How loudly the noise-classifier branch argues. 0.0 switches it off, which is
# the experiment that answers whether it helps at all.

SPECTRAL_LOSS_WEIGHT = 200.0
# How loudly the compressed-magnitude grader argues, against SI-SNR's 1.0.
# WHY IT EXISTS: oracle_ceiling.py showed the model reaches 73% of the best
# SI-SNR physically achievable but only 23% of the best PESQ, which is what
# happens when the thing you train is not the thing you are judged on. SI-SNR
# grades waveform overlap, dominated by loud vowels and by the phase we copy
# from the input and never change. This term grades the magnitude spectrum
# bin by bin, which is exactly what the mask controls.
# HIGHER leans toward PESQ and STOI; 0.0 reproduces every run up to 0009.
# WHY 200: the two terms are on wildly different scales, so this was measured
# rather than guessed. `--loss-scales` backpropagates each term alone and
# compares the size of the update it produces; the weight that makes them pull
# equally hard is 381 at random init and 187 on run 0009's trained weights.
# 200 sits in that range. Sensible sweep: 50 (SI-SNR still leads) to 600
# (spectral leads). Do not reason about this from the printed loss VALUES --
# their ratio swings from 8 to 451 over a run and means nothing.

SPECTRAL_COMPRESSION = 0.3
# Power the magnitudes are raised to before comparing. 1.0 means raw power,
# where one loud vowel outweighs a hundred consonants; LOWER flattens the
# range so quiet consonants -- the parts that carry intelligibility -- count.
# 0.3 is the usual choice and roughly matches how loudness is perceived.
# Sensible range: 0.15 to 0.5. Below ~0.1 near-silent bins start to dominate.

GRAD_CLIP = 5.0
# Caps how far one step can move the weights. Guards against a single strange
# batch destroying hours of training. 0 disables. Sensible range: 1-10.

EVAL_EVERY = 5
# How often to run the expensive perceptual metrics and save audio samples.
# Cheap metrics run every epoch regardless. Sensible range: 1-10.

EARLY_STOP_PATIENCE = 0
# Stop if validation loss has not improved for this many epochs. 0 disables it.
# DISABLED deliberately. Early stopping watches validation LOSS, but the results
# are judged on PESQ, and the two diverge: every run stopped while its PESQ was
# still climbing, several on their very last evaluation. That made training
# LENGTH the hidden variable behind comparisons meant to isolate something else,
# because configurations that plateau later in loss get more epochs. A fixed
# budget for every run removes the confound.

SEED = 0
# Fixed so two runs with the same settings are comparable.

STEPS_PER_EPOCH = 80
# Batches per epoch, held FIXED regardless of how much data is in use. This
# exists for the data-scaling experiment: if an epoch were simply "one pass",
# a quarter of the data would also mean a quarter of the gradient updates, and
# a worse result would say nothing about whether diversity or training amount
# caused it. Sampling with replacement to a fixed step count isolates the one
# variable being tested. 80 matches a full pass at batch 16.

SAMPLE_CLIPS = 4
# How many fixed validation clips to save audio and mask pictures for at each
# evaluation. Always the same clips, chosen to span easy and hard conditions,
# so listening across epochs is an honest comparison.


# ═══════════════════════════════════════════════════════════════════
#  LOCKED — set by the data and the model, not experiment settings
# ═══════════════════════════════════════════════════════════════════

SAMPLE_RATE = model_crn.SAMPLE_RATE
CLIP_SAMPLES = model_crn.CLIP_SAMPLES
EPS = 1e-8


def build_crn(**over):
    return model_crn.CRN(
        channels=over.get("encoder_channels", model_crn.ENCODER_CHANNELS),
        lookahead=over.get("lookahead_frames", model_crn.LOOKAHEAD_FRAMES),
        hidden=over.get("lstm_hidden", model_crn.LSTM_HIDDEN),
        layers=over.get("lstm_layers", model_crn.LSTM_LAYERS),
        mask_max=over.get("mask_max", model_crn.MASK_MAX))


# The registry. A new architecture is one entry here plus its model file.
ARCHITECTURES = {"crn": build_crn}


# --------------------------------------------------------------------------
# losses
# --------------------------------------------------------------------------

def si_snr(est, ref):
    """Scale-invariant signal-to-noise ratio, in dB, per item.

    Scale-invariant means a quieter-but-correct answer is not punished: the
    reference is first projected onto the estimate, so only the *shape* of the
    waveform is judged, not its volume. Higher is better, so the loss below
    negates it.
    """
    est = est - est.mean(dim=-1, keepdim=True)
    ref = ref - ref.mean(dim=-1, keepdim=True)
    alpha = (est * ref).sum(-1, keepdim=True) / (ref.pow(2).sum(-1, keepdim=True) + EPS)
    target = alpha * ref
    noise = est - target
    return 10 * torch.log10((target.pow(2).sum(-1) + EPS) / (noise.pow(2).sum(-1) + EPS))


def si_snr_loss(est, ref):
    return -si_snr(est, ref).mean()


def spectral_loss(est, ref, power=SPECTRAL_COMPRESSION):
    """Compressed-magnitude distance between two waveforms.

    Both signals are turned into spectrograms and compared bin by bin: for
    every pitch band at every instant, is there the right amount of sound
    here? That is precisely the question the mask answers, which is why this
    grader can steer the mask where SI-SNR cannot.

    The compression is the important part. Raw magnitudes span several orders
    of magnitude, so a squared error on them is decided almost entirely by the
    loudest vowels; the fricatives and stops that make speech *intelligible*
    contribute almost nothing. Raising both sides to 0.3 first pulls the range
    together so quiet bins carry real weight -- the same trick the ear plays,
    and the reason this correlates with PESQ and STOI far better than SI-SNR.

    Phase is deliberately ignored here. The model does not predict phase, so
    grading it would only add noise to the gradient.
    """
    est_mag = model_crn.to_spectrogram(est).abs()
    ref_mag = model_crn.to_spectrogram(ref).abs()
    est_c = (est_mag + EPS) ** power
    ref_c = (ref_mag + EPS) ** power
    return (est_c - ref_c).pow(2).mean()


def loss_scale_report(model, loader, device, n_batches=4):
    """Print how hard each loss term pulls, so the weight is a measurement.

    Two terms added together only share influence if they push with comparable
    force. Comparing their VALUES is misleading: SI-SNR is in dB and sits near
    -11, a squared error on compressed magnitudes near 0.02, and the ratio
    between them swings by 50x over the course of training. What decides who
    actually steers the model is the size of the GRADIENT each term produces,
    so that is what this measures -- backpropagating each term on its own and
    reporting the norm of the resulting update.

    The balancing weight it prints makes the two pull equally hard. That is a
    starting point, not an answer: the useful setting is found by sweeping
    around it.
    """
    # train mode, not eval: cuDNN refuses to backpropagate through an LSTM in
    # eval mode. Nothing is optimised here and the process exits straight
    # after, so the batch-norm running statistics this touches do not matter.
    model.train()
    rows = []
    for i, (noisy, clean, _) in enumerate(loader):
        if i >= n_batches:
            break
        noisy, clean = noisy.to(device), clean.to(device)
        est, _, _, _ = model(noisy)
        terms = {"si_snr": si_snr_loss(est, clean),
                 "spectral": spectral_loss(est, clean)}
        out = {}
        for name, term in terms.items():
            model.zero_grad(set_to_none=True)
            term.backward(retain_graph=True)
            total = sum(float(p.grad.pow(2).sum()) for p in model.parameters()
                        if p.grad is not None)
            out[name] = (float(term.detach()), total ** 0.5)
        rows.append(out)
    model.zero_grad(set_to_none=True)

    print(f"{'term':<12}{'value':>12}{'grad norm':>14}")
    grads = {}
    for name in ("si_snr", "spectral"):
        v = float(np.mean([r[name][0] for r in rows]))
        g = float(np.mean([r[name][1] for r in rows]))
        grads[name] = g
        print(f"{name:<12}{v:>12.5f}{g:>14.5f}")
    w = grads["si_snr"] / max(grads["spectral"], EPS)
    print(f"\nweight that makes them pull equally hard: {w:.1f}")
    return grads


# --------------------------------------------------------------------------
# one pass
# --------------------------------------------------------------------------

def run_epoch(model, loader, device, optimizer=None, aux_weight=AUX_LOSS_WEIGHT,
              spec_weight=SPECTRAL_LOSS_WEIGHT):
    """One pass. Trains if an optimizer is given, otherwise evaluates."""
    training = optimizer is not None
    model.train(training)

    totals = {"loss": 0.0, "audio_loss": 0.0, "spec_loss": 0.0, "aux_loss": 0.0,
              "si_snr": 0.0, "aux_correct": 0.0, "n": 0}
    ce = nn.CrossEntropyLoss()

    for noisy, clean, label in loader:
        noisy = noisy.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        with torch.set_grad_enabled(training):
            est, _, logits, _ = model(noisy)
            audio_loss = si_snr_loss(est, clean)
            spec_loss = spectral_loss(est, clean) if spec_weight \
                else torch.zeros((), device=device)
            aux_loss = ce(logits, label) if (logits is not None and aux_weight) \
                else torch.zeros((), device=device)
            loss = audio_loss + spec_weight * spec_loss + aux_weight * aux_loss

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if GRAD_CLIP:
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

        n = noisy.shape[0]
        totals["n"] += n
        # detach before converting: these are only for reporting, and keeping
        # them attached to the graph risks holding the whole graph alive
        totals["loss"] += loss.detach().item() * n
        totals["audio_loss"] += audio_loss.detach().item() * n
        totals["spec_loss"] += spec_loss.detach().item() * n
        totals["aux_loss"] += aux_loss.detach().item() * n
        with torch.no_grad():
            totals["si_snr"] += float(si_snr(est, clean).sum())
            if logits is not None:
                totals["aux_correct"] += float((logits.argmax(1) == label).sum())

    n = max(totals["n"], 1)
    return {"loss": totals["loss"] / n,
            "audio_loss": totals["audio_loss"] / n,
            "spec_loss": totals["spec_loss"] / n,
            "aux_loss": totals["aux_loss"] / n,
            "si_snr": totals["si_snr"] / n,
            "aux_accuracy": totals["aux_correct"] / n}


def measure_inference_ms(model, device):
    """Milliseconds to process one 3-second clip. Catches speed regressions."""
    model.eval()
    x = torch.randn(1, CLIP_SAMPLES, device=device)
    with torch.no_grad():
        for _ in range(3):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
    return (time.perf_counter() - t0) / 10 * 1000


def pick_sample_clips(dataset, k=SAMPLE_CLIPS):
    """Fixed clips spanning easy and hard, so listening is a fair comparison.

    Chosen once and never re-rolled: hearing the same clip improve across
    epochs is the point, and a different clip each time would destroy it.
    """
    rows = [(i, dataset.describe(i)) for i in range(len(dataset))]
    impulsive = {"assault_rifle", "heavy_artillery", "mortar", "rocket"}
    hard = [r for r in rows if r[1]["category"] in impulsive]
    easy = [r for r in rows if r[1]["category"] not in impulsive]
    hard.sort(key=lambda r: r[1]["snr_db"])
    easy.sort(key=lambda r: r[1]["snr_db"])
    picked = []
    if hard:
        picked += [hard[0][0], hard[-1][0]]          # worst and best impulsive
    if easy:
        picked += [easy[0][0], easy[-1][0]]          # worst and best steady
    seen, out = set(), []
    for i in picked:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out[:k]


def save_samples(model, dataset, indices, device, run, epoch):
    """Write enhanced audio and a mask picture for the fixed clips."""
    import soundfile as sf
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()
    written = []
    for idx in indices:
        m = dataset.describe(idx)
        x = torch.from_numpy(m["noisy"]).unsqueeze(0).to(device)
        with torch.no_grad():
            est, mask, _, _ = model(x)
        est = est.squeeze(0).cpu().numpy()
        mk = mask.squeeze(0).cpu().numpy().T                 # (freq, time)

        stem = f"e{epoch:04d}_{m['category']}_{idx:03d}"
        sf.write(run.sample_path(stem + "_enhanced.wav"), est, SAMPLE_RATE,
                 subtype="PCM_16")
        # the noisy and clean sides never change, so write them once
        for kind in ("noisy", "clean"):
            p = run.sample_path(f"ref_{m['category']}_{idx:03d}_{kind}.wav")
            if not os.path.exists(p):
                sf.write(p, m[kind], SAMPLE_RATE, subtype="PCM_16")

        fig, ax = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
        noisy_spec = model_crn.to_spectrogram(
            torch.from_numpy(m["noisy"]).unsqueeze(0)).abs().squeeze(0).numpy().T
        db = 20 * np.log10(np.maximum(noisy_spec, 1e-8))
        ax[0].imshow(db, origin="lower", aspect="auto", cmap="magma",
                     vmin=db.max() - 80, vmax=db.max(),
                     extent=[0, 3, 0, SAMPLE_RATE / 2000])
        ax[0].set_title(f"noisy input - {m['category']}, "
                        f"SNR {m['snr_db']:+.1f} dB", fontsize=9, loc="left")
        ax[0].set_ylabel("kHz", fontsize=8)
        im = ax[1].imshow(mk, origin="lower", aspect="auto", cmap="viridis",
                          vmin=0, vmax=1, extent=[0, 3, 0, SAMPLE_RATE / 2000])
        ax[1].set_title(f"predicted mask, epoch {epoch} - bright means keep",
                        fontsize=9, loc="left")
        ax[1].set_ylabel("kHz", fontsize=8)
        ax[1].set_xlabel("seconds", fontsize=8)
        fig.colorbar(im, ax=ax[1], fraction=0.03)
        fig.tight_layout()
        fig.savefig(run.sample_path(stem + "_mask.png"), dpi=90)
        plt.close(fig)
        written.append(stem)
    return written


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="crn", choices=sorted(ARCHITECTURES))
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=LEARNING_RATE)
    ap.add_argument("--aux-weight", type=float, default=AUX_LOSS_WEIGHT)
    ap.add_argument("--spec-weight", type=float, default=SPECTRAL_LOSS_WEIGHT,
                    help="weight on the compressed-magnitude loss; 0 "
                         "reproduces the SI-SNR-only runs up to 0009")
    ap.add_argument("--loss-scales", action="store_true",
                    help="print the size of each loss term and exit")
    ap.add_argument("--mask-max", type=float, default=model_crn.MASK_MAX,
                    help="mask ceiling; 1.0 attenuate only, >1 allows boosting")
    ap.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    ap.add_argument("--note", default=None,
                    help="why you are running this; saved to notes.md")
    ap.add_argument("--resume", default=None, metavar="RUN_NAME")
    ap.add_argument("--data-fraction", type=float, default=1.0,
                    help="train on this fraction of the speech clips, with "
                         "the step count held fixed")
    ap.add_argument("--limit-batches", type=int, default=None,
                    help="stop each epoch after N batches (smoke testing)")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set = SpeechNoiseMixtures("train")            # fresh noise each epoch
    if args.data_fraction < 1.0:
        keep = max(1, int(len(train_set.speech) * args.data_fraction))
        # take an evenly spaced slice, not the first N: the manifest is sorted
        # by language, so the head would be Assamese and English only
        step = len(train_set.speech) / keep
        train_set.speech = [train_set.speech[int(i * step)] for i in range(keep)]
    val_set = FrozenMixtures("validation")              # fixed, comparable
    classes = read_noise_classes()

    from torch.utils.data import RandomSampler
    sampler = RandomSampler(train_set, replacement=True,
                            num_samples=STEPS_PER_EPOCH * args.batch_size)
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              sampler=sampler, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=0)

    model = ARCHITECTURES[args.arch](mask_max=args.mask_max).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    if args.loss_scales:
        # Reports the size of each term so the weight can be chosen from a
        # measurement. With --resume it reports at that run's operating point,
        # which is the balance that actually matters; without, at random init.
        if args.resume:
            ck = torch.load(RunLog.open(args.resume).checkpoint_path("best.pt"),
                            map_location=device, weights_only=False)
            model.load_state_dict(ck["model"])
            print(f"loaded {args.resume} best.pt (epoch {ck['epoch']})")
        loss_scale_report(model, val_loader, device)
        return

    opts = {"adamw": torch.optim.AdamW, "adam": torch.optim.Adam,
            "rmsprop": torch.optim.RMSprop}
    kw = {"lr": args.lr}
    if OPTIMIZER in ("adamw", "adam"):
        kw["weight_decay"] = WEIGHT_DECAY
    optimizer = opts[OPTIMIZER](model.parameters(), **kw)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5)

    config = {
        "arch": args.arch, "epochs": args.epochs, "batch_size": args.batch_size,
        "data_fraction": args.data_fraction, "steps_per_epoch": STEPS_PER_EPOCH,
        "learning_rate": args.lr, "weight_decay": WEIGHT_DECAY,
        "optimizer": OPTIMIZER, "aux_loss_weight": args.aux_weight,
        "spectral_loss_weight": args.spec_weight,
        "spectral_compression": SPECTRAL_COMPRESSION,
        "grad_clip": GRAD_CLIP, "eval_every": args.eval_every,
        "early_stop_patience": EARLY_STOP_PATIENCE, "seed": SEED,
        "n_fft": model_crn.N_FFT, "hop_length": model_crn.HOP_LENGTH,
        "lookahead_frames": model_crn.LOOKAHEAD_FRAMES,
        "lstm_hidden": model_crn.LSTM_HIDDEN,
        "lstm_layers": model_crn.LSTM_LAYERS,
        "encoder_channels": list(model_crn.ENCODER_CHANNELS),
        "mask_max": args.mask_max,
        "parameters": n_params, "device": str(device),
        "train_clips": len(train_set), "val_clips": len(val_set),
        "classes": classes,
        "lookahead_ms": model_crn.TOTAL_LOOKAHEAD_MS,
        "chunk_ms": model_crn.CHUNK_SAMPLES / SAMPLE_RATE * 1000,
    }

    start_epoch, best_val, patience = 1, float("inf"), 0
    if args.resume:
        run = RunLog.open(args.resume)
        ck = torch.load(run.checkpoint_path("last.pt"), map_location=device,
                        weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        start_epoch = ck["epoch"] + 1
        best_val = ck.get("best_val", float("inf"))
        print(f"resumed {run.name} at epoch {start_epoch}")
    else:
        run = RunLog.create(args.arch, config, notes=args.note,
                            tag_keys=("spectral_loss_weight", "aux_loss_weight"))

    infer_ms = measure_inference_ms(model, device)
    run.update_config(inference_ms_per_clip=infer_ms,
                      total_latency_ms=model_crn.TOTAL_LOOKAHEAD_MS
                      + config["chunk_ms"]
                      + infer_ms * config["chunk_ms"] / 3000.0)

    sample_idx = pick_sample_clips(val_set)

    print(f"run           {run.name}")
    print(f"architecture  {args.arch}, {n_params:,} parameters on {device}")
    print(f"data          {len(train_set)} train (fresh mixtures), "
          f"{len(val_set)} validation (frozen)")
    print(f"latency       {model_crn.TOTAL_LOOKAHEAD_MS:.0f} ms lookahead + "
          f"100 ms chunk + {infer_ms * 100 / 3000:.2f} ms compute")
    print(f"samples       clips {sample_idx} tracked every {args.eval_every} "
          f"epochs\n")
    print(f"{'ep':>4} {'train':>9} {'val':>9} {'si-snr':>8} {'aux':>6} "
          f"{'lr':>9} {'sec':>6}")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.perf_counter()
        if args.limit_batches:
            from itertools import islice
            limited = list(islice(iter(train_loader), args.limit_batches))
            tr = run_epoch(model, limited, device, optimizer, args.aux_weight,
                           args.spec_weight)
        else:
            tr = run_epoch(model, train_loader, device, optimizer,
                           args.aux_weight, args.spec_weight)
        va = run_epoch(model, val_loader, device, None, args.aux_weight,
                       args.spec_weight)
        secs = time.perf_counter() - t0
        scheduler.step(va["loss"])
        lr_now = optimizer.param_groups[0]["lr"]

        row = {"epoch": epoch,
               "train_loss": tr["loss"], "val_loss": va["loss"],
               "train_si_snr": tr["si_snr"], "val_si_snr": va["si_snr"],
               "train_spec_loss": tr["spec_loss"],
               "val_spec_loss": va["spec_loss"],
               "train_aux_loss": tr["aux_loss"], "val_aux_loss": va["aux_loss"],
               "train_aux_accuracy": tr["aux_accuracy"],
               "val_aux_accuracy": va["aux_accuracy"],
               "learning_rate": lr_now, "epoch_seconds": secs,
               "inference_ms": measure_inference_ms(model, device)}
        if device.type == "cuda":
            row["gpu_gb"] = torch.cuda.max_memory_allocated() / 1024 ** 3

        improved = va["loss"] < best_val - 1e-6
        if improved:
            best_val, patience = va["loss"], 0
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "config": run.config(), "val_loss": best_val},
                       run.checkpoint_path("best.pt"))
        else:
            patience += 1
        row["best_val_loss"] = best_val
        row["epochs_since_improvement"] = patience

        flag = " *" if improved else ""
        print(f"{epoch:>4} {tr['loss']:>9.4f} {va['loss']:>9.4f} "
              f"{va['si_snr']:>8.2f} {va['aux_accuracy']:>6.2f} "
              f"{lr_now:>9.2e} {secs:>6.1f}{flag}")

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            row["samples"] = save_samples(model, val_set, sample_idx, device,
                                          run, epoch)
            # the expensive perceptual metrics, and the noisy baseline they
            # only mean anything against
            import evaluate as ev
            res, _ = ev.evaluate(model, val_set, device)
            o = res["overall"]
            for k in ("si_snr", "stoi", "pesq"):
                row[f"{k}_enhanced"] = o[f"{k}_enhanced"]
                row[f"{k}_noisy"] = o[f"{k}_noisy"]
                row[f"{k}_gain"] = o[f"{k}_gain"]
            row["eval_aux_accuracy"] = res["aux_accuracy"]
            row["by_category"] = res["by_category"]
            row["targets_met"] = res["targets_met"]
            print(f"     eval  PESQ {o['pesq_noisy']:.2f} -> "
                  f"{o['pesq_enhanced']:.2f}   "
                  f"STOI {o['stoi_noisy']:.3f} -> {o['stoi_enhanced']:.3f}   "
                  f"SNR {o['si_snr_noisy']:+.1f} -> "
                  f"{o['si_snr_enhanced']:+.1f} dB")

        torch.save({"model": model.state_dict(),
                    "optimizer": optimizer.state_dict(), "epoch": epoch,
                    "best_val": best_val}, run.checkpoint_path("last.pt"))
        run.log(**row)


        if EARLY_STOP_PATIENCE and patience >= EARLY_STOP_PATIENCE:
            print(f"\nstopping: validation loss has not improved for "
                  f"{patience} epochs")
            break

    print(f"\nbest validation loss {best_val:.4f}")
    print(f"run saved to runs/{run.name}/")
    print(f"evaluate it with:  python scripts/evaluate.py {run.name}")


if __name__ == "__main__":
    main()
