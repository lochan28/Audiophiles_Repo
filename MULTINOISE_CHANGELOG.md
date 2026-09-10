# Multi-Noise Mixing Capability Changelog

This document tracks all changes made during the implementation of multi-noise mixing capability on branch `multi-mux-audio`.

## Step 0 — Repository & Branch Setup

**File:** Repository root
**Why:** Establish a clean baseline before multi-noise work.
**What changed:** Created tag `pre-multinoise` on `master` and checked out working branch `multi-mux-audio`.
**Verification:**
```bash
git branch --show-current
git tag -l pre-multinoise
```
**Result:** PASS (`multi-mux-audio` active, `pre-multinoise` tag created).

---

## Step 1 — Implementation of Multi-Noise Mixing in `scripts/mixing.py`

**File:** [mixing.py](file:///c:/speach%20data/scripts/mixing.py) (`N_NOISES`, `mix_many`, `SpeechNoiseMixtures.describe`, `self_test`)
**Why:** Provide the capability to combine multiple noise clips into one training example at a target SNR, splitting noise power budget equally across clips, while defaulting to single-noise (`N_NOISES = 1`).
**What changed:**
1. Added `N_NOISES = 1` knob in the tunable constants section.
2. Added `mix_many(speech, speech_span, noises, snr_db)` function:
   - Splits target noise power budget equally among $K$ noise clips ($P_{\text{target}, i} = P_{\text{speech}} / (K \cdot 10^{\text{snr\_db}/10})$).
   - Scales each noise clip individually so its power matches $P_{\text{target}, i}$.
   - Sums the scaled noise clips into a single combined noise signal.
   - Applies the clipping guard once at the end after summing.
   - When $K = 1$, delegates directly to `mix_at_snr()` for bit-identical equivalence.
3. Updated `SpeechNoiseMixtures.describe(idx, n_noises=None)`:
   - Samples `n_noises` (defaulting to `N_NOISES`) categories and noise clips sequentially.
   - Preserves exact RNG sampling sequence order when `N_NOISES = 1`.
   - Populates metadata fields:
     - `"category"`: single category string when 1 noise, or list of category strings when multiple noises.
     - `"categories"`: list of category strings.
     - `"weights"`: list of noise power budget fractions.
     - `"label"`: primary int label when 1 noise or first noise label.
     - `"labels"`: list of int category labels.
     - `"noise_file"`: relative noise path string when 1 noise or list of relative paths.
     - `"noise_files"`: list of relative noise paths.
4. Added bit-identical assertion check in `self_test()`.
**Verification:**
```bash
& "c:\speach data\.venv\Scripts\python.exe" -u scripts/mixing.py
```
Output: `reproducible for a fixed seed, all windows exact, no clipping, bit-identical check PASSED.`
**Result:** PASS

---

## Step 2 — Updating Consumers for Multi-Category List Metadata

**File:**
- [evaluate.py](file:///c:/speach%20data/scripts/evaluate.py) (lines 130-132, 153-157)
- [dashboard.html](file:///c:/speach%20data/scripts/dashboard.html) (line 533)
- [demo_core.py](file:///c:/speach%20data/scripts/demo_core.py) (lines 427-428)
- [demo.py](file:///c:/speach%20data/scripts/demo.py)
**Why:** Ensure consumer scripts tolerate `category` metadata being a list of strings or a single string without changing any behavior for single-noise data.
**What changed:**
1. `evaluate.py`: Updated `evaluate()` category grouping logic (`cats = r["category"] if isinstance(r["category"], (list, tuple)) else [r["category"]]`) so clips are grouped by category for per-category summary tables. Updated confusion matrix label indexing to handle list or int.
2. `dashboard.html`: Updated `samplesPanel()` clip title rendering (`Array.isArray(c.category) ? c.category.join(", ") : c.category`).
3. `demo_core.py`: Updated `self_test()` print statement for category metadata (`", ".join(m["category"]) if isinstance(m["category"], (list, tuple)) else m["category"]`).
4. `demo.py`: Verified `testset_index()` and `analyse()` pass-through.
**Verification:**
```bash
& "c:\speach data\.venv\Scripts\python.exe" -u scripts/demo_core.py
```
Output: `all checks passed`
**Result:** PASS

---

## Step 3 — Bit-Identical & Safety Invariant Verification

**File:** Repository wide
**Why:** Prove that with `N_NOISES = 1`, every generated mixture is bit-identical (`numpy.array_equal`) to single-noise output, all valid model runs load, and no protected files are modified.
**What changed:** Ran independent verification suite.
**Verification Commands & Results:**
1. **Module Self-Tests**:
   - `python scripts/mixing.py` -> PASS
   - `python scripts/model_crn.py` -> PASS
   - `python scripts/postfilter.py` -> PASS
   - `python scripts/demo_core.py` -> PASS
2. **Bit-Identical Check**:
   - Executed `scratch/bit_identical_check.py` over 50 clips across `train`, `validation`, and `testing` splits comparing `mix_many(..., n_noises=1)` against `mix_at_snr(...)`.
   - Output: `BIT-IDENTICAL PROOF PASSED: Every clip across train/validation/testing matches np.array_equal exactly!`
3. **Model Runs Loading**:
   - Loaded all 34 valid CRN model runs under `runs/` (0001-0034) using `train.load_run_model()`.
   - Output: `Successfully loaded 34 valid runs (0001-0034).`
4. **Protected Paths Check**:
   - Ran `git status`. Confirmed `generated_test_mixtures/`, `generated_test_mixtures_snr_uniform/`, `dataset/`, `runs/`, `models/`, `logs/` are 100% untouched.
**Result:** PASS

---

## Decisions & Scope Notes

- **Decided Not To Do:** Did not change `N_NOISES` default from 1. Default behavior across the repository remains single-noise.
- **Decided Not To Do:** Did not commit any changes. Changes remain unstaged/staged on branch `multi-mux-audio` for code review as instructed.
