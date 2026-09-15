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
import mixing
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
# How loudly the auxiliary branch argues. 0.0 switches it off. What the branch
# is asked to predict is model_crn.AUX_TARGET; this is only how much it weighs.
# Size it with --loss-scales rather than by eye -- the printed VALUES of two
# loss terms say nothing about which one actually steers the model.

SPECTRAL_LOSS_WEIGHT = 600.0
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
# 200 sits in that range. Do not reason about this from the printed loss
# VALUES -- their ratio swings from 8 to 451 over a run and means nothing.
# SWEPT, and the knob turned out to be nearly flat: 50 -> 1.780, 200 -> 1.784,
# 600 -> 1.799, against 1.682 with the term switched off. Almost all of the
# gain comes from having the term AT ALL, and twelve times the weight buys a
# fifth as much again. 600 is the default because it was best and costs
# nothing, but there is no further run hiding in this knob.

MASK_LOSS_WEIGHT = 0.0
# How loudly a fourth grader argues: one that marks the MASK itself against the
# mask the model should have drawn, rather than judging the audio that came out.
# WHY IT EXISTS: the mask is the only thing this model produces, and until now
# nothing ever told it which parts of the mask were wrong -- only that the
# finished audio was off, leaving it to work backwards. `mask_diagnosis.py`
# shows the mask correlates with the ideal one at just 0.617 and regresses
# toward the middle: where the ideal wants 0 it leaves 0.134, where the ideal
# wants 1 it keeps 0.707. That hedging is optimal under uncertainty and cannot
# be undone afterwards -- multiplying, powering and contrast-expanding the mask
# all make things monotonically worse -- so the signal has to change instead.
# The target is computable during training because the clean speech is on hand:
# ideal = |clean| / |noisy|, capped at the model's own mask ceiling. It is a
# teaching aid only and costs nothing at deployment, like the noise classifier.
# 0.0 is OFF, which reproduces runs 0010-0018. Size it with `--loss-scales`,
# which reports the gradient each term produces; never by comparing loss values.
# MEASURED: the weight that makes this term pull as hard as SI-SNR is 112 at
# random initialisation and 60 on run 0016's trained weights, so 100 sits in
# that range and is what the first experiment uses. For reference the spectral
# term's equal-pull weight is ~190 and it runs at 600, i.e. about 3x, so a mask
# weight of 100 gives this term roughly a third of the spectral term's say.
# Sensible sweep after the first result: 30 and 300.

SPECTRAL_RESOLUTIONS = (512,)
# STFT window sizes the compressed-magnitude grader is computed at, in samples.
# The scores are AVERAGED, not summed, so adding resolutions does not silently
# multiply this term's weight against SI-SNR -- see the note in spectral_loss.
#
# MEASURED AND REJECTED. Run 0029 used (256, 512, 1024) against the three-seed
# control 0026/0027/0028 and lost on every metric at once:
#
#       PESQ  2.1077 -> 2.0752   -4.6 sigma
#       STOI  0.8859 -> 0.8837   -5.4 sigma
#     SI-SNR  13.271 -> 13.154   -4.7 sigma
#
# The idea was that one window length forces a trade -- short windows say
# precisely WHEN a sound happened but are vague about its PITCH, long windows
# do the reverse -- and our noise splits along exactly that line, gunfire and
# mortar on one side, rotor and engine hum on the other.
#
# The likely reason it failed anyway: the model emits one gain per bin on the
# 512/160 grid, so an error the 512 lens reports has a knob that fixes it,
# while an error seen only at 256 or 1024 is smeared across bins the mask
# controls separately. Averaging cut the actionable feedback to a third of its
# strength and replaced the rest with feedback the operator cannot cleanly act
# on. This is a hypothesis, not a measurement.
#
# Multi-resolution spectral loss is standard practice in speech enhancement.
# It is the third borrowed technique to lose here, after finer LSTM bands
# (run 0013) and direct mask supervision (run 0019). Do not retry it without
# a reason the above explanation does not already cover.
#
# The machinery is kept because it costs nothing at (512,) -- one lens, one
# STFT pair, byte-identical to the pre-0029 loss (verified 0.26467431 both
# ways). Anyone re-testing this should change the tuple, not the code.

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
# Starting point for the random number generators.
#
# IT DOES NOT MAKE RUNS REPRODUCIBLE. This was assumed for sixteen runs and is
# false: runs 0014 and 0016 share a configuration and this seed, and diverge at
# epoch 1 (training loss 25.1153 against 25.6876) and by 0.029 PESQ over epochs
# 45-80. cuDNN's LSTM backward is nondeterministic and `torch.manual_seed` does
# not contain it.
#
# The consequence is a methodological one. A within-run standard deviation
# across evaluations is 0.005 to 0.010 PESQ, but that measures evaluation
# wobble, NOT how much two runs of the same settings differ. Comparing two
# single runs against the within-run figure will call noise a result. Use
# `--seed` to repeat a configuration and compare distributions instead.

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
        mask_max=over.get("mask_max", model_crn.MASK_MAX),
        freq_strides=over.get("freq_strides", model_crn.ENCODER_FREQ_STRIDES),
        aux_target=over.get("aux_target", model_crn.AUX_TARGET))


# The registry. A new architecture is one entry here plus its model file.
ARCHITECTURES = {"crn": build_crn}


# Which saved checkpoint to reach for, best first.
#
# best_pesq.pt before best.pt is the whole point: `best.pt` is chosen by
# validation LOSS, and loss and PESQ diverge badly. On run 0014 the best-loss
# epoch was 26 scoring PESQ 1.703 while the best-PESQ epoch was 65 scoring
# 1.832, so deploying `best.pt` would have thrown away 0.13 PESQ that had
# already been paid for. last.pt comes next because a converged run's final
# weights are near its plateau, which is still better than an early stop.
CHECKPOINT_PREFERENCE = ("best_pesq.pt", "last.pt", "best.pt")


def build_from_config(cfg, device):
    """Rebuild a model exactly as the recorded run built it.

    Reading every shape from the run's own config, rather than from the module
    constants, is what makes an old run loadable after the defaults move on.
    """
    channels = cfg.get("encoder_channels", model_crn.ENCODER_CHANNELS)
    # Runs before 0013 predate ENCODER_FREQ_STRIDES and all halved at every
    # layer; falling back to the current default would build the wrong shape.
    strides = cfg.get("encoder_freq_strides", [2] * len(channels))
    return ARCHITECTURES[cfg.get("arch", "crn")](
        encoder_channels=channels,
        lstm_hidden=cfg.get("lstm_hidden", model_crn.LSTM_HIDDEN),
        lstm_layers=cfg.get("lstm_layers", model_crn.LSTM_LAYERS),
        lookahead_frames=cfg.get("lookahead_frames",
                                 model_crn.LOOKAHEAD_FRAMES),
        mask_max=cfg.get("mask_max", model_crn.MASK_MAX),
        # Runs before 0030 predate AUX_TARGET and all had the 7-way category
        # head; falling back to the current default would build a 1-output
        # head and refuse to load their checkpoints.
        aux_target=cfg.get("aux_target", "category"),
        freq_strides=strides).to(device)


def load_run_model(run, device, checkpoint="auto"):
    """Load a run's weights. Returns (model, checkpoint name, epoch).

    "auto" walks CHECKPOINT_PREFERENCE and takes the first that exists, so
    older runs without a best_pesq.pt still load something sensible.
    """
    names = (CHECKPOINT_PREFERENCE if checkpoint == "auto"
             else (checkpoint if checkpoint.endswith(".pt")
                   else checkpoint + ".pt",))
    for name in names:
        path = run.checkpoint_path(name)
        if os.path.exists(path):
            model = build_from_config(run.config(), device)
            ck = torch.load(path, map_location=device, weights_only=False)
            model.load_state_dict(ck["model"])
            return model.eval(), name, ck.get("epoch")
    raise FileNotFoundError(
        f"no checkpoint in {run.name}; looked for {', '.join(names)}")


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


def _stft_mag(wav, n_fft, hop):
    """Magnitude spectrogram at an arbitrary resolution.

    `model_crn.to_spectrogram` is hard-wired to the model's own N_FFT and hop
    because the network depends on that exact framing. The loss does not: it
    only compares two waveforms, so it is free to look at them through other
    windows.

    The hop is scaled to keep the model's overlap ratio (160/512), so the 512
    entry here is IDENTICAL to what to_spectrogram produces. That matters: it
    makes adding resolutions a purely additive change rather than one that also
    quietly alters the term we already have.
    """
    window = torch.hann_window(n_fft, device=wav.device)
    spec = torch.stft(wav, n_fft, hop, n_fft, window,
                      center=True, return_complex=True)
    return spec.abs()


def spectral_loss(est, ref, power=SPECTRAL_COMPRESSION,
                  resolutions=SPECTRAL_RESOLUTIONS):
    """Compressed-magnitude distance, averaged over several STFT resolutions.

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

    AVERAGED rather than summed across resolutions. Summing would triple this
    term's pull against SI-SNR when three windows are used, which would confound
    "more lenses" with "three times the weight" -- and the weight has already
    been swept, so that confound would be unreadable. Averaging keeps
    SPECTRAL_LOSS_WEIGHT meaning what it meant before.

    Phase is deliberately ignored here. The model does not predict phase, so
    grading it would only add noise to the gradient.
    """
    ratio = model_crn.HOP_LENGTH / model_crn.N_FFT
    total = 0.0
    for n_fft in resolutions:
        hop = max(1, int(round(n_fft * ratio)))
        est_c = (_stft_mag(est, n_fft, hop) + EPS) ** power
        ref_c = (_stft_mag(ref, n_fft, hop) + EPS) ** power
        total = total + (est_c - ref_c).pow(2).mean()
    return total / len(resolutions)


def local_snr_target(noisy, clean):
    """How much of each 10 ms frame is speech, as a number in [0, 1].

    The mixture is built as speech + scaled noise, and both halves are rescaled
    together if the sum would clip, so `noisy - clean` recovers the noise
    EXACTLY. Verified on the frozen validation split: the SNR implied by that
    residual matches the recorded SNR to two decimals. Nothing has to be
    plumbed through the data pipeline to get this.

    Per frame, speech energy / (speech + noise energy). Two reasons for this
    form rather than SNR in decibels:

    - Decibels are unbounded and useless here. Measured over the validation
      split the per-frame value runs from -140 dB to +140 dB, because a frame
      of pure silence divides by nothing. A squared error on that is decided
      entirely by frames where the answer is arbitrary.
    - This ratio is the frame-level version of exactly what the mask computes
      per bin, and it needs no invented clamping constants.

    0 means the frame is all noise, 1 means it is all speech, 0.5 is 0 dB.
    """
    n_spec = model_crn.to_spectrogram(noisy - clean).abs().pow(2).sum(-1)
    s_spec = model_crn.to_spectrogram(clean).abs().pow(2).sum(-1)
    return s_spec / (s_spec + n_spec + EPS)          # (B, T)


def aux_term(model, logits, noisy, clean, label):
    """The auxiliary loss, and a readable score, for whichever target is set.

    Returns (loss, score, score_name). The score is only for reporting: how
    often the category is right, or the mean absolute error in speech-presence.
    """
    if model.aux_target == "local_snr":
        target = local_snr_target(noisy, clean)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, target)
        # Cross-entropy rather than squared error because the target is a
        # probability and the head ends in a sigmoid: the pair has a clean
        # gradient of (prediction - target), where MSE through a sigmoid goes
        # flat exactly where the target spends 47% of its time, at the ends.
        with torch.no_grad():
            score = float((torch.sigmoid(logits) - target).abs().mean())
        return loss, score, "mae"
    loss = nn.functional.cross_entropy(logits, label)
    with torch.no_grad():
        score = float((logits.argmax(1) == label).float().mean())
    return loss, score, "accuracy"


def ideal_mask(noisy, clean, mask_max):
    """The mask the model should have drawn, from audio it never sees.

    |clean| / |noisy| per bin, capped at the model's own ceiling so the target
    is something the architecture can actually produce -- asking a sigmoid
    bounded at 1.0 to output 1.4 would be teaching it to fail.
    """
    mag_n = model_crn.to_spectrogram(noisy).abs()
    mag_c = model_crn.to_spectrogram(clean).abs()
    return (mag_c / (mag_n + EPS)).clamp(max=mask_max)


def mask_loss(mask, noisy, clean, mask_max):
    """Squared error between the drawn mask and the ideal one."""
    return (mask - ideal_mask(noisy, clean, mask_max)).pow(2).mean()


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
    for i, (noisy, clean, label) in enumerate(loader):
        if i >= n_batches:
            break
        noisy, clean = noisy.to(device), clean.to(device)
        est, mask, logits, _ = model(noisy)
        terms = {"si_snr": si_snr_loss(est, clean),
                 "spectral": spectral_loss(est, clean),
                 "mask": mask_loss(mask, noisy, clean,
                                   model.decoder.mask_max),
                 "aux": aux_term(model, logits, noisy, clean,
                                 label.to(device))[0]}
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
    for name in ("si_snr", "spectral", "mask", "aux"):
        v = float(np.mean([r[name][0] for r in rows]))
        g = float(np.mean([r[name][1] for r in rows]))
        grads[name] = g
        print(f"{name:<12}{v:>12.5f}{g:>14.5f}")
    for name in ("spectral", "mask", "aux"):
        w = grads["si_snr"] / max(grads[name], EPS)
        print(f"weight that makes {name} pull as hard as si_snr: {w:.1f}")
    return grads


# --------------------------------------------------------------------------
# one pass
# --------------------------------------------------------------------------

def run_epoch(model, loader, device, optimizer=None, aux_weight=AUX_LOSS_WEIGHT,
              spec_weight=SPECTRAL_LOSS_WEIGHT,
              mask_weight=MASK_LOSS_WEIGHT,
              compression=SPECTRAL_COMPRESSION):
    """One pass. Trains if an optimizer is given, otherwise evaluates."""
    training = optimizer is not None
    model.train(training)

    totals = {"loss": 0.0, "audio_loss": 0.0, "spec_loss": 0.0,
              "mask_loss": 0.0, "aux_loss": 0.0,
              "si_snr": 0.0, "aux_score": 0.0, "n": 0}
    score_name = "accuracy"

    for noisy, clean, label in loader:
        noisy = noisy.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        with torch.set_grad_enabled(training):
            est, mask, logits, _ = model(noisy)
            audio_loss = si_snr_loss(est, clean)
            spec_loss = (spectral_loss(est, clean, compression)
                         if spec_weight
                         else torch.zeros((), device=device))
            mloss = (mask_loss(mask, noisy, clean, model.decoder.mask_max)
                     if mask_weight
                     else torch.zeros((), device=device))
            if logits is not None and aux_weight:
                aux_loss, aux_score, score_name = aux_term(
                    model, logits, noisy, clean, label)
            else:
                aux_loss = torch.zeros((), device=device)
                aux_score = 0.0
            loss = (audio_loss + spec_weight * spec_loss
                    + mask_weight * mloss + aux_weight * aux_loss)

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
        totals["mask_loss"] += mloss.detach().item() * n
        totals["aux_loss"] += aux_loss.detach().item() * n
        with torch.no_grad():
            totals["si_snr"] += float(si_snr(est, clean).sum())
        totals["aux_score"] += aux_score * n

    n = max(totals["n"], 1)
    return {"loss": totals["loss"] / n,
            "audio_loss": totals["audio_loss"] / n,
            "spec_loss": totals["spec_loss"] / n,
            "mask_loss": totals["mask_loss"] / n,
            "aux_loss": totals["aux_loss"] / n,
            "si_snr": totals["si_snr"] / n,
            # One of these is always None: the two targets are not comparable
            # and merging them into one column would make old and new runs
            # look like they were measuring the same thing.
            "aux_accuracy": (totals["aux_score"] / n
                             if score_name == "accuracy" else None),
            "aux_mae": (totals["aux_score"] / n
                        if score_name == "mae" else None)}


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
    ap.add_argument("--aux-target", default=model_crn.AUX_TARGET,
                    choices=("category", "local_snr"),
                    help="what the auxiliary branch predicts")
    ap.add_argument("--spec-weight", type=float, default=SPECTRAL_LOSS_WEIGHT,
                    help="weight on the compressed-magnitude loss; 0 "
                         "reproduces the SI-SNR-only runs up to 0009")
    ap.add_argument("--lookahead", type=int,
                    default=model_crn.LOOKAHEAD_FRAMES,
                    help="frames of the future the encoder may see; 10 ms each, hard cap 10 (2 per encoder layer)")
    ap.add_argument("--compression", type=float,
                    default=SPECTRAL_COMPRESSION,
                    help="power the magnitudes are raised to before comparing; lower makes quiet consonants count more")
    ap.add_argument("--mask-weight", type=float,
                    default=MASK_LOSS_WEIGHT,
                    help="weight on the mask-vs-ideal-mask loss; 0 reproduces runs 0010-0018")
    ap.add_argument("--loss-scales", action="store_true",
                    help="print the size of each loss term and exit")
    ap.add_argument("--lstm-hidden", type=int, default=model_crn.LSTM_HIDDEN,
                    help="width of the memory stage")
    ap.add_argument("--seed", type=int, default=SEED,
                    help="repeat a configuration under a different seed, to "
                         "measure how much two identical runs differ")
    ap.add_argument("--channel-scale", type=float, default=1.0,
                    help="multiply every encoder layer's width by this; 1.5 "
                         "reproduces the capacity of run 0002")
    ap.add_argument("--mask-max", type=float, default=model_crn.MASK_MAX,
                    help="mask ceiling; 1.0 attenuate only, >1 allows boosting")
    ap.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    ap.add_argument("--note", default=None,
                    help="why you are running this; saved to notes.md")
    ap.add_argument("--resume", default=None, metavar="RUN_NAME")
    ap.add_argument("--init-from", default=None, metavar="RUN_NAME",
                    help="start a NEW run from another run's trained weights, "
                         "with a fresh optimizer and learning rate. Unlike "
                         "--resume this never writes into the source run.")
    ap.add_argument("--data-fraction", type=float, default=1.0,
                    help="train on this fraction of the speech clips, with "
                         "the step count held fixed")
    ap.add_argument("--limit-batches", type=int, default=None,
                    help="stop each epoch after N batches (smoke testing)")
    ap.add_argument("--noise-root", default=None, metavar="DIR",
                    help="read noise from DIR instead of dataset/ (e.g. "
                         "dataset_v2) -- speech clips still come from "
                         "dataset/ either way, only the noise set moves")
    ap.add_argument("--multi-noise", action="store_true",
                    help="use the v2 mixing rules: 1-4 noise clips summed "
                         "per mixture (weighted, see NOISE_COUNT_WEIGHTS in "
                         "mixing.py) instead of exactly one")
    ap.add_argument("--eval-mixtures-root", default=None, metavar="DIR",
                    help="read the frozen validation/testing mixtures from "
                         "DIR instead of generated_test_mixtures/")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.noise_root:
        mixing.NOISE_DIR = os.path.abspath(args.noise_root)

    train_set = SpeechNoiseMixtures("train", multi_noise=args.multi_noise)
    if args.data_fraction < 1.0:
        keep = max(1, int(len(train_set.speech) * args.data_fraction))
        # take an evenly spaced slice, not the first N: the manifest is sorted
        # by language, so the head would be Assamese and English only
        step = len(train_set.speech) / keep
        train_set.speech = [train_set.speech[int(i * step)] for i in range(keep)]
    val_set = FrozenMixtures("validation", root=args.eval_mixtures_root)  # fixed, comparable
    classes = read_noise_classes()

    from torch.utils.data import RandomSampler
    sampler = RandomSampler(train_set, replacement=True,
                            num_samples=STEPS_PER_EPOCH * args.batch_size)
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              sampler=sampler, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=0)

    channels = tuple(max(1, round(c * args.channel_scale))
                     for c in model_crn.ENCODER_CHANNELS)
    model = ARCHITECTURES[args.arch](mask_max=args.mask_max,
                                     encoder_channels=channels,
                                     lstm_hidden=args.lstm_hidden,
                                     aux_target=args.aux_target,
                                     lookahead_frames=args.lookahead).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    # Fine-tuning. Deliberately separate from --resume, which continues a run
    # IN PLACE and would append epochs to that run's own metrics and overwrite
    # its checkpoints -- correct for finishing an interrupted run, destructive
    # if what you wanted was a new experiment starting from a good model.
    #
    # Note that a fine-tune needs its own CONTROL to be readable. A converged
    # run has wound its learning rate down to almost nothing, so waking it up
    # at a fresh rate moves the model whatever else you change; runs 0033 and
    # 0034 are the worked example of that pairing.
    if args.init_from:
        src = RunLog.open(args.init_from)
        loaded, ck_name, ck_epoch = load_run_model(src, device)
        model.load_state_dict(loaded.state_dict())
        print(f"initialised from {src.name}, {ck_name} epoch {ck_epoch}")
        del loaded

    if args.loss_scales:
        # Reports how hard each term pulls, so its weight can be chosen
        # from a measurement. With --resume it reports at that run's
        # operating point, which is the balance that actually matters;
        # without, at random initialisation.
        if args.resume:
            model, ck_name, ep = load_run_model(
                RunLog.open(args.resume), device)
            print(f"loaded {ck_name} (epoch {ep})")
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
        "aux_target": args.aux_target,
        "init_from": args.init_from,
        "spectral_loss_weight": args.spec_weight,
        "mask_loss_weight": args.mask_weight,
        "spectral_compression": args.compression,
        "spectral_resolutions": list(SPECTRAL_RESOLUTIONS),
        # Tagged so a run name says how many lenses the grader used.
        "n_resolutions": len(SPECTRAL_RESOLUTIONS),
        "grad_clip": GRAD_CLIP, "eval_every": args.eval_every,
        "early_stop_patience": EARLY_STOP_PATIENCE, "seed": args.seed,
        "n_fft": model_crn.N_FFT, "hop_length": model_crn.HOP_LENGTH,
        "lookahead_frames": args.lookahead,
        # From the arguments, not the module constants: these can be
        # overridden per run and the record has to say what actually ran.
        "lstm_hidden": args.lstm_hidden,
        "lstm_layers": model_crn.LSTM_LAYERS,
        "encoder_channels": list(channels),
        "channel_scale": args.channel_scale,
        "encoder_freq_strides": list(model_crn.ENCODER_FREQ_STRIDES),
        # How many frequency bands the LSTM actually sees. The single number
        # that says whether the recurrent core can resolve a harmonic comb.
        "lstm_bands": model.encoder.out_freq,
        "mask_max": args.mask_max,
        "parameters": n_params, "device": str(device),
        "train_clips": len(train_set), "val_clips": len(val_set),
        "classes": classes,
        # Which SNR distribution produced this run's training data AND
        # the frozen set it is scored on. Runs with different profiles
        # are NOT comparable: the 90/10 change alone moves the headline
        # numbers by about +0.13 PESQ with the model untouched.
        "snr_profile": mixing.SNR_PROFILE,
        "lookahead_ms": model_crn.STFT_LOOKAHEAD_MS + args.lookahead * 10.0,
        "chunk_ms": model_crn.CHUNK_SAMPLES / SAMPLE_RATE * 1000,
    }

    start_epoch, best_val, patience = 1, float("inf"), 0
    best_pesq = float("-inf")
    if args.resume:
        run = RunLog.open(args.resume)
        ck = torch.load(run.checkpoint_path("last.pt"), map_location=device,
                        weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        start_epoch = ck["epoch"] + 1
        best_val = ck.get("best_val", float("inf"))
        best_pesq = ck.get("best_pesq", float("-inf"))
        print(f"resumed {run.name} at epoch {start_epoch}")
    else:
        run = RunLog.create(args.arch, config, notes=args.note,
                            tag_keys=("aux_target", "spectral_compression",
                                      "n_resolutions",
                                      "spectral_loss_weight",
                                      "lookahead_frames", "seed"))

    infer_ms = measure_inference_ms(model, device)
    run.update_config(inference_ms_per_clip=infer_ms,
                      total_latency_ms=config["lookahead_ms"]
                      + config["chunk_ms"]
                      + infer_ms * config["chunk_ms"] / 3000.0)

    sample_idx = pick_sample_clips(val_set)

    print(f"run           {run.name}")
    print(f"architecture  {args.arch}, {n_params:,} parameters on {device}")
    print(f"data          {len(train_set)} train (fresh mixtures), "
          f"{len(val_set)} validation (frozen)")
    print(f"latency       {config['lookahead_ms']:.0f} ms lookahead + "
          f"100 ms chunk + {infer_ms * 100 / 3000:.2f} ms compute")
    print(f"samples       clips {sample_idx} tracked every {args.eval_every} "
          f"epochs\n")
    aux_col = "mae" if args.aux_target == "local_snr" else "acc"
    print(f"{'ep':>4} {'train':>9} {'val':>9} {'si-snr':>8} {aux_col:>6} "
          f"{'lr':>9} {'sec':>6}")

    # A run that ends short should say why in its own config, so a later
    # reader does not have to guess whether 34 of 80 epochs means "stopped by
    # hand", "early stopping fired" or "it crashed".
    stopped_early = None
    last_epoch = start_epoch - 1

    # Clear a stop left over from a previous run of this directory, otherwise
    # resuming would halt again immediately for a reason nobody remembers.
    if run.stop_requested():
        run.clear_stop()

    # Blank while training, filled in when the loop ends. The dashboard reads
    # this to tell "still going" from "already finished": looks_running() is a
    # wall-clock heuristic and stays true for minutes after a run has actually
    # stopped, which would otherwise leave a live Stop button on a dead run.
    run.update_config(stopped_early=None, epochs_completed=None)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.perf_counter()
        if args.limit_batches:
            from itertools import islice
            limited = list(islice(iter(train_loader), args.limit_batches))
            tr = run_epoch(model, limited, device, optimizer, args.aux_weight,
                           args.spec_weight, args.mask_weight, args.compression)
        else:
            tr = run_epoch(model, train_loader, device, optimizer,
                           args.aux_weight, args.spec_weight, args.mask_weight, args.compression)
        va = run_epoch(model, val_loader, device, None, args.aux_weight,
                       args.spec_weight, args.mask_weight, args.compression)
        secs = time.perf_counter() - t0
        scheduler.step(va["loss"])
        lr_now = optimizer.param_groups[0]["lr"]

        row = {"epoch": epoch,
               "train_loss": tr["loss"], "val_loss": va["loss"],
               "train_si_snr": tr["si_snr"], "val_si_snr": va["si_snr"],
               "train_spec_loss": tr["spec_loss"],
               "val_spec_loss": va["spec_loss"],
               "train_mask_loss": tr["mask_loss"],
               "val_mask_loss": va["mask_loss"],
               "train_aux_loss": tr["aux_loss"], "val_aux_loss": va["aux_loss"],
               "train_aux_accuracy": tr["aux_accuracy"],
               "val_aux_accuracy": va["aux_accuracy"],
               "train_aux_mae": tr["aux_mae"], "val_aux_mae": va["aux_mae"],
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
              f"{va['si_snr']:>8.2f} "
              f"{(va['aux_mae'] if va['aux_mae'] is not None else va['aux_accuracy']):>6.2f} "
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

            # Keep the best model BY QUALITY, separately from best.pt which is
            # kept by validation loss. The two disagree by tens of epochs and
            # over 0.1 PESQ, and it is this one that should be deployed and
            # evaluated. See CHECKPOINT_PREFERENCE.
            if o["pesq_enhanced"] is not None and o["pesq_enhanced"] > best_pesq:
                best_pesq = o["pesq_enhanced"]
                torch.save({"model": model.state_dict(), "epoch": epoch,
                            "config": run.config(), "pesq": best_pesq},
                           run.checkpoint_path("best_pesq.pt"))
                print(f"           new best PESQ {best_pesq:.4f}, "
                      f"saved best_pesq.pt")
            row["best_pesq"] = best_pesq
            print(f"     eval  PESQ {o['pesq_noisy']:.2f} -> "
                  f"{o['pesq_enhanced']:.2f}   "
                  f"STOI {o['stoi_noisy']:.3f} -> {o['stoi_enhanced']:.3f}   "
                  f"SNR {o['si_snr_noisy']:+.1f} -> "
                  f"{o['si_snr_enhanced']:+.1f} dB")

        torch.save({"model": model.state_dict(),
                    "optimizer": optimizer.state_dict(), "epoch": epoch,
                    "best_val": best_val, "best_pesq": best_pesq},
                   run.checkpoint_path("last.pt"))
        run.log(**row)
        last_epoch = epoch

        # Asked to stop from the dashboard. This is checked here, at the very
        # end of the epoch, because best.pt, best_pesq.pt, last.pt and this
        # epoch's metrics row are all already on disk by this point -- so
        # stopping costs nothing. The run ends in exactly the state it would
        # have been in had this been its last planned epoch, and it can still
        # be resumed later from last.pt.
        if run.stop_requested():
            req = run.stop_request() or {}
            why = req.get("reason") or ""
            print(f"\nstopping: requested from the dashboard"
                  + (f" -- {why}" if why else ""))
            stopped_early = "manual stop from the dashboard" \
                + (f": {why}" if why else "")
            run.clear_stop()
            break

        if EARLY_STOP_PATIENCE and patience >= EARLY_STOP_PATIENCE:
            print(f"\nstopping: validation loss has not improved for "
                  f"{patience} epochs")
            stopped_early = (f"early stopping: validation loss flat for "
                             f"{patience} epochs")
            break

    run.update_config(epochs_completed=last_epoch, stopped_early=stopped_early)

    print(f"\nbest validation loss {best_val:.4f}")
    print(f"run saved to runs/{run.name}/")
    print(f"evaluate it with:  python scripts/evaluate.py {run.name}")


if __name__ == "__main__":
    main()
