"""Does the model already know where the speech is, without being taught?

Runs 0030 and 0031 added a head that predicts, for every 10 ms frame, how much
of that frame is speech. The head learned the task very well (MAE 0.074 and
0.095 against 0.370 for the best fixed guess) and the audio did not improve at
all. The natural reading is that the information was ALREADY in the model's
shared representation, so the head taught it nothing -- it merely read out
something the model had worked out on its own in order to paint the mask.

That reading is an inference from a run whose trunk still received a sliver of
auxiliary gradient. This measures it directly instead, on run 0026, which was
trained with the old CATEGORY head and never saw a local-SNR target in its
life. The model is FROZEN; only the little readout on top is fitted. If the
readout succeeds, the knowledge was there all along.

THE CONTROL THAT MAKES THIS MEAN ANYTHING
-----------------------------------------
A good score on its own proves little, because it might just say that local
SNR is easy to read off almost any features of a spectrogram. So the identical
probe is also fitted to a RANDOMLY INITIALISED model of the same shape. The
gap between the two is what training actually contributed. Without that
control the headline number is uninterpretable.
"""

import os
import sys
import argparse

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import model_crn
import train as T
from runlog import RunLog
from mixing import SpeechNoiseMixtures, FrozenMixtures
from torch.utils.data import DataLoader

# ══════════════════════════════════════════════════════════════════
#  TUNING KNOBS -- safe to change while exploring
# ══════════════════════════════════════════════════════════════════

PROBE_EPOCHS = 8
# Passes over the fitting data. The probe is tiny and the features are fixed,
# so it converges fast; raise this only if the two probes are still improving
# when the run ends, which the printed per-epoch line will show.

PROBE_LR = 1e-3
# Adam step size for the readout only. The model itself is frozen.

BATCH_SIZE = 16
# Clips per step. Only affects speed and memory, not the conclusion.

FIT_BATCHES = 60
# Batches of fresh training-split mixtures used to fit the probe, per epoch.
# 60 x 16 = 960 clips, about 289,000 frames, against 16,449 parameters in the
# larger probe -- comfortably more data than parameters.

PROBE_KINDS = ("linear", "mlp")
# Which readouts to fit.
#   "linear"  one matrix. The strict test: it can only find information that
#             is laid out plainly, so success here is strong evidence.
#   "mlp"     Linear-ELU-Linear, IDENTICAL to the head runs 0030 and 0031
#             trained with, so its score is directly comparable to their
#             0.074 and 0.095.

# ══════════════════════════════════════════════════════════════════
#  FIXED -- these mirror the runs being probed
# ══════════════════════════════════════════════════════════════════

RUN = "0026_crn_comp03_spec600_la6_s0"
HIDDEN_UNITS = 64          # the aux head's middle layer, as trained


def make_probe(kind, hidden):
    if kind == "linear":
        return nn.Linear(hidden, 1)
    return nn.Sequential(nn.Linear(hidden, HIDDEN_UNITS), nn.ELU(),
                         nn.Linear(HIDDEN_UNITS, 1))


@torch.no_grad()
def features(model, noisy):
    """The model's shared representation: (B, T, hidden), frozen."""
    x, _ = model.encoder(model_crn.to_spectrogram(noisy).abs())
    _, seq, _ = model.memory(x)
    return seq


def fit_and_score(model, kind, fit_loader, val, device, label):
    """Fit one readout on frozen features, report MAE on frozen validation."""
    probe = make_probe(kind, model.memory.hidden).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=PROBE_LR)
    n_par = sum(p.numel() for p in probe.parameters())

    for epoch in range(PROBE_EPOCHS):
        for i, (noisy, clean, _) in enumerate(fit_loader):
            if i >= FIT_BATCHES:
                break
            noisy, clean = noisy.to(device), clean.to(device)
            target = T.local_snr_target(noisy, clean)
            pred = probe(features(model, noisy)).squeeze(-1)
            loss = nn.functional.binary_cross_entropy_with_logits(pred, target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        mae = evaluate_probe(model, probe, val, device)
        print(f"    {label:<22} {kind:<7} epoch {epoch + 1}/{PROBE_EPOCHS}"
              f"   val MAE {mae:.4f}")
    return mae, n_par


@torch.no_grad()
def evaluate_probe(model, probe, val, device):
    err = n = 0.0
    for noisy, clean, _ in val:
        noisy, clean = noisy.to(device), clean.to(device)
        target = T.local_snr_target(noisy, clean)
        pred = torch.sigmoid(probe(features(model, noisy)).squeeze(-1))
        err += float((pred - target).abs().sum())
        n += target.numel()
    return err / n


def baselines(val, device):
    """What a probe that has learned nothing would score."""
    per_clip, allv = [], []
    for noisy, clean, _ in val:
        t = T.local_snr_target(noisy.to(device), clean.to(device))
        per_clip.append(float((t - t.mean(dim=1, keepdim=True)).abs().mean()))
        allv.append(t.flatten().cpu().numpy())
    v = np.concatenate(allv)
    const = min(float(np.abs(v - k).mean()) for k in np.linspace(0, 1, 201))
    return const, float(np.mean(per_clip))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=RUN)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = RunLog.open(args.run)
    trained, ck, epoch = T.load_run_model(run, device)
    trained.eval()
    print(f"probing {args.run}, {ck} epoch {epoch}")
    print(f"aux target it was TRAINED with: {trained.aux_target}\n")

    # The control: same architecture, same shapes, no training at all.
    random_model = T.build_from_config(run.config(), device).eval()

    fit_loader = DataLoader(SpeechNoiseMixtures("train"),
                            batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val = list(DataLoader(FrozenMixtures("validation"),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=0))

    const, clip_mean = baselines(val, device)
    results = {}
    for label, model in (("trained (run 0026)", trained),
                         ("random init", random_model)):
        for kind in PROBE_KINDS:
            mae, n_par = fit_and_score(model, kind, fit_loader, val,
                                       device, label)
            results[(label, kind)] = (mae, n_par)
        print()

    print(f"\n{'readout on frozen features':<36}{'params':>9}{'val MAE':>10}")
    print("-" * 55)
    print(f"{'guess one constant forever':<36}{'-':>9}{const:>10.4f}")
    print(f"{'guess each clip average':<36}{'-':>9}{clip_mean:>10.4f}")
    for (label, kind), (mae, n_par) in results.items():
        print(f"{label + ', ' + kind:<36}{n_par:>9,}{mae:>10.4f}")
    print("-" * 55)
    print(f"{'head TRAINED in, run 0030 (w=43)':<36}{'-':>9}{0.074:>10.4f}")
    print(f"{'head TRAINED in, run 0031 (w=0.53)':<36}{'-':>9}{0.095:>10.4f}")
    print("\nIf the trained rows land near the run 0030/0031 figures while the")
    print("random rows sit near the baselines, the model had already worked")
    print("out where the speech is and the auxiliary head taught it nothing.")


if __name__ == "__main__":
    main()
