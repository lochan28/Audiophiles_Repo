ERB hybrid resolution, 65 low bins + 64 ERB bands (GTCRN). Same recipe as run 0026; the ideal-mask ceiling with this constraint is 3.388 vs 3.389, so any quality change is the model, not the representation. The point is the Raspberry Pi: LSTM input 1728 -> 960.

---

## Verdict (2026-09-12)

Converged mean over epochs >= 110 (9 evals), against the three-seed control
(runs 0026/0027/0028, mean 2.1077 / 0.8859 / 13.271, floor sd 0.0061 /
0.0004 / 0.0218):

    PESQ    2.1018   -0.0059   -0.8 sigma   holds
    STOI    0.8836   -0.0023   -5.0 sigma   regresses
    SI-SNR  13.194   -0.077    -3.1 sigma   regresses

REJECTED as a free change: two metrics fall outside the noise floor. The
ideal-mask ceiling (scripts/erb_ceiling.py) said the representation costs
nothing; the trained model pays anyway, because the same change halves the
LSTM's bands from 9 to 5 as a side effect, and this project already measured
that LSTM band count matters (run 0013). The two cannot be separated without
giving back the compute win.

What it buys, x86 streaming, stage 1, 4 threads:

    fp32   12.1 -> 8.6 ms   (1.4x)
    int8   10.4 -> 8.2 ms   (1.27x)

UNDECIDED as a Raspberry Pi trade, and the deciding measurement needs a
board. The interesting case is the Pi 5 two-stage system, which sat at
110.8 ms int8 -- 10% over the deadline. If this brings it under 100 ms, the
post-filter's +0.05 PESQ comes back, ten times what this run gave up, and the
pair could pass the acceptance rule on net. The post-filter would need
retraining against this run's output first (scripts/postfilter_test.py).

Branch erb-hybrid, unmerged. HANDOFF.md section 13 carries the decision test.
