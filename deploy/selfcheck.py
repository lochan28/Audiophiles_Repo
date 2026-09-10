"""
selfcheck.py - Pi 5 Deployment Check
Runs offline to verify the bundle is working correctly.
"""

import os
import sys
import time
import platform
import subprocess

# Add bundle to path so we can import its modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'bundle'))

import torch
import numpy as np

# ═══════════════════════════════════════════════════════════════════
#  TUNING KNOBS — safe to change, this is what you experiment with
# ═══════════════════════════════════════════════════════════════════

RUN_DIR = os.path.join(os.path.dirname(__file__), 'bundle', 'runs', '0026_crn_comp03_spec600_la6_s0')
# Path to the trained run directory containing config.json and checkpoints.

POSTFILTER_WEIGHTS = os.path.join(os.path.dirname(__file__), 'bundle', 'models', 'postfilter_ch48_la0_snr25.pt')
# Path to the specific post-filter weights to load. 

CLIPS_DIR = os.path.join(os.path.dirname(__file__), 'bundle', 'generated_test_mixtures', 'testing')
# Directory containing 'clean' and 'noisy' folders with the test clips.

N_THREADS_TO_TEST = (1, 2, 4)
# Tuple of thread counts to benchmark inference latency against.

REF_PESQ = 1.9826
REF_STOI = 0.8660
REF_SI_SNR = 14.019
# Reference metrics from the x86 machine to compare against.

# ═══════════════════════════════════════════════════════════════════

def get_cpu_model():
    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":")[1].strip()
        except Exception:
            pass
    elif platform.system() == "Windows":
        try:
            return subprocess.check_output(["wmic", "cpu", "get", "name"]).decode().strip().split('\n')[1].strip()
        except Exception:
            pass
    return platform.processor()

def si_snr(est, ref):
    eps = 1e-8
    est = est - est.mean()
    ref = ref - ref.mean()
    alpha = np.dot(est, ref) / (np.dot(ref, ref) + eps)
    target = alpha * ref
    noise = est - target
    return float(10 * np.log10((np.sum(target ** 2) + eps) / (np.sum(noise ** 2) + eps)))

def main():
    print(f"Python version: {platform.python_version()}")
    print(f"Torch version: {torch.__version__}")
    print(f"Initial thread count: {torch.get_num_threads()}")
    print(f"CPU model: {get_cpu_model()}")
    print("-" * 50)

    try:
        import model_crn
        import postfilter
        import runlog
        import train
        import soundfile as sf
    except ImportError as e:
        print(f"Failed to import bundle modules: {e}")
        print("NO-GO: Bundle Import Failed")
        return

    # Check pesq/pystoi
    try:
        from pesq import pesq as pesq_score
        from pystoi import stoi as stoi_score
    except ImportError as e:
        print(f"Warning: Offline package missing ({e}). Cannot score PESQ/STOI.")
        pesq_score = None
        stoi_score = None

    # 1. Self-tests
    print("Running self-tests...")
    try:
        model_crn.self_test()
        postfilter.self_test()
        print("Self-tests passed.")
    except Exception as e:
        print(f"Self-tests failed: {e}")
        print("NO-GO: Self-tests failed")
        return

    print("-" * 50)

    # 2. Load models
    print("Loading models...")
    device = torch.device('cpu')
    try:
        run = runlog.RunLog(RUN_DIR)
        model, ck_name, epoch = train.load_run_model(run, device, "auto")
        print(f"Loaded CRN: {ck_name} epoch {epoch}")
        
        pf_model, _ = postfilter.load(POSTFILTER_WEIGHTS, device)
        pf_model.eval()
        print("Loaded Postfilter.")
    except Exception as e:
        print(f"Failed to load models: {e}")
        print("NO-GO: Model Load Failed")
        return

    print("-" * 50)

    # 3. Score clips
    print("Scoring clips...")
    noisy_dir = os.path.join(CLIPS_DIR, 'noisy')
    clean_dir = os.path.join(CLIPS_DIR, 'clean')
    
    if not os.path.exists(noisy_dir) or not os.path.exists(clean_dir):
        print("Clip directories missing.")
        print("NO-GO: Missing Clips")
        return

    clips = sorted([f for f in os.listdir(noisy_dir) if f.endswith('.wav')])
    
    pesqs, stois, snrs = [], [], []
    
    for clip in clips:
        noisy_wav, _ = sf.read(os.path.join(noisy_dir, clip), dtype='float32')
        clean_wav, _ = sf.read(os.path.join(clean_dir, clip), dtype='float32')
        
        noisy_pt = torch.from_numpy(noisy_wav).unsqueeze(0).to(device)
        
        with torch.no_grad():
            est, mask, _, _ = model(noisy_pt)
            final_est_pt = postfilter.apply(pf_model, noisy_pt, mask)
        
        final_est = final_est_pt.squeeze(0).numpy()
        
        snr_val = si_snr(final_est, clean_wav)
        snrs.append(snr_val)
        
        if pesq_score and stoi_score:
            try:
                p_val = pesq_score(16000, clean_wav, final_est, 'wb')
                pesqs.append(p_val)
            except:
                pass
            try:
                s_val = stoi_score(clean_wav, final_est, 16000, extended=False)
                stois.append(s_val)
            except:
                pass
                
    avg_snr = np.mean(snrs)
    avg_pesq = np.mean(pesqs) if pesqs else 0.0
    avg_stoi = np.mean(stois) if stois else 0.0
    
    print(f"Average SI-SNR: {avg_snr:.3f} (Diff to Ref: {avg_snr - REF_SI_SNR:+.3f})")
    if pesq_score and stoi_score:
        print(f"Average PESQ:   {avg_pesq:.4f} (Diff to Ref: {avg_pesq - REF_PESQ:+.4f})")
        print(f"Average STOI:   {avg_stoi:.4f} (Diff to Ref: {avg_stoi - REF_STOI:+.4f})")
    else:
        print("Skipping PESQ/STOI evaluation because packages are missing.")

    print("-" * 50)
    
    # 4. Benchmark Timing
    print("Benchmarking...")
    dummy_wav = torch.randn(1, 48000)
    
    for threads in N_THREADS_TO_TEST:
        torch.set_num_threads(threads)
        times = []
        
        with torch.no_grad():
            for _ in range(3):
                e, m, _, _ = model(dummy_wav)
                postfilter.apply(pf_model, dummy_wav, m)
                
        for _ in range(10):
            t0 = time.perf_counter()
            with torch.no_grad():
                e, m, _, _ = model(dummy_wav)
                postfilter.apply(pf_model, dummy_wav, m)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
            
        median_ms = np.median(times)
        worst_ms = np.max(times)
        print(f"Threads: {threads} | Median: {median_ms:.1f} ms | Worst: {worst_ms:.1f} ms")

    print("-" * 50)
    
    if not (pesq_score and stoi_score):
        print("NO-GO: missing pesq/pystoi packages")
    else:
        print("GO")


if __name__ == '__main__':
    main()
