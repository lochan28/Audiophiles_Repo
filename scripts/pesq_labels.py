"""Real PESQ scores, computed off the training thread.

The critic has to be taught by something, and the only honest teacher is PESQ
itself. That is affordable or not depending entirely on this file.

THE ARITHMETIC THAT DECIDES IT

PESQ takes 82 ms per clip on one core. A training epoch here is 17.3 s. Scoring
all 1292 training clips serially costs 106 s, so labelling every clip every
epoch would make the run six times slower -- a 50 minute experiment becomes
five hours, which is not a thing anyone will run twice.

But PESQ is pure CPU work and training is pure GPU work, and this machine has
16 logical cores. Spread across them the same job costs about 9 s, which fits
inside one epoch. So the cost is not "six times slower", it is "free, if the
two are actually overlapped". Hence submit(), which hands the work to the pool
and returns immediately, rather than only a blocking score().

WHY THE WORKERS NEVER SEE TORCH

Windows has no fork: every worker process re-imports this module from scratch.
If torch were imported at the top, spawning 14 workers would mean 14 fresh
torch imports, several seconds each, every time a pool is created. Tensors are
therefore converted to numpy in the parent, and this module stays light.
"""

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ══════════════════════════════════════════════════════════════════
#  TUNING KNOBS -- safe to change while exploring
# ══════════════════════════════════════════════════════════════════

WORKERS = max(1, (os.cpu_count() or 4) - 2)
# How many processes score PESQ in parallel. Two cores are deliberately left
# free: one for the main process feeding the GPU and one for the operating
# system. Taking all 16 makes PESQ finish slightly sooner and the training
# loop stall waiting for data, which is a net loss.

CHUNK_SIZE = 8
# Clips handed to a worker at a time. Each hand-off costs a pickle round trip,
# so tiny chunks spend their time on postage; huge chunks leave workers idle at
# the end of a batch. 8 is a middle setting and is not sensitive.

# ══════════════════════════════════════════════════════════════════
#  FIXED -- set by the data on disk
# ══════════════════════════════════════════════════════════════════

SAMPLE_RATE = 16000


def _score_one(pair):
    """One (clean, other) pair. Runs inside a worker process.

    Imports live here rather than at module scope so that a worker pays for the
    pesq extension only, and only once.
    """
    from pesq import pesq as pesq_score
    from pesq import NoUtterancesError
    clean, other = pair
    try:
        return float(pesq_score(SAMPLE_RATE, clean, other, "wb"))
    except (NoUtterancesError, ValueError, RuntimeError):
        # PESQ refuses signals it cannot align or that hold no detectable
        # speech. None keeps that clip out of the critic's training set rather
        # than teaching it a fabricated score.
        return None


def _as_numpy(x):
    """Tensors or arrays in, contiguous float32 numpy out, always on the CPU."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.ascontiguousarray(np.asarray(x, dtype=np.float32))


class PesqPool:
    """A pool of worker processes that score PESQ.

    Use as a context manager so the workers are always shut down:

        with PesqPool() as pool:
            future = pool.submit(clean_batch, enhanced_batch)
            ...   # train on the GPU while PESQ runs on the CPU
            scores = future.result()
    """

    def __init__(self, workers=WORKERS):
        self.workers = workers
        self._pool = None

    def __enter__(self):
        self._pool = ProcessPoolExecutor(max_workers=self.workers)
        # One thread in THIS process whose only job is to wait for the workers.
        # The work itself happens in the worker processes, so this thread holds
        # the GIL for nothing but inter-process bookkeeping and does not slow
        # the training loop down. It exists purely so submit() can return
        # straight away instead of blocking on the whole batch.
        self._driver = ThreadPoolExecutor(max_workers=1)
        return self

    def __exit__(self, *exc):
        self._driver.shutdown(wait=True)
        self._pool.shutdown(wait=True)
        self._driver = self._pool = None
        return False

    def submit(self, clean, other):
        """Start scoring. Returns a future holding a list of scores or Nones.

        Call this BEFORE the GPU work it should overlap with, not after. The
        conversion to numpy happens here, on the calling thread, so the caller
        cannot hand over tensors it is about to overwrite.
        """
        pairs = [(_as_numpy(c), _as_numpy(o)) for c, o in zip(clean, other)]
        return self._driver.submit(self._map, pairs)

    def _map(self, pairs):
        return list(self._pool.map(_score_one, pairs, chunksize=CHUNK_SIZE))

    def score(self, clean, other):
        """Blocking version, for code that has nothing else to get on with."""
        pairs = [(_as_numpy(c), _as_numpy(o)) for c, o in zip(clean, other)]
        return list(self._pool.map(_score_one, pairs, chunksize=CHUNK_SIZE))


def self_test():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from mixing import FrozenMixtures
    from evaluate import safe_pesq

    n = 64
    ds = FrozenMixtures("validation")
    rows = [ds.describe(i) for i in range(n)]
    clean = [r["clean"] for r in rows]
    noisy = [r["noisy"] for r in rows]

    print(f"scoring {n} clips, {WORKERS} workers on {os.cpu_count()} cores\n")

    t0 = time.perf_counter()
    serial = [safe_pesq(c, o) for c, o in zip(clean, noisy)]
    t_serial = time.perf_counter() - t0

    with PesqPool() as pool:
        t0 = time.perf_counter()
        parallel = pool.score(clean, noisy)
        t_par = time.perf_counter() - t0

        # A pool that returns different numbers is worse than no pool at all,
        # so this is checked rather than assumed.
        both = [(a, b) for a, b in zip(serial, parallel)]
        worst = max(abs(a - b) for a, b in both if a is not None
                    and b is not None)
        assert worst == 0.0, f"pool disagrees with serial by {worst}"
        assert [a is None for a in serial] == [b is None for b in parallel]
        print(f"agreement     identical to serial on all {n} clips")

        t0 = time.perf_counter()
        fut = pool.submit(clean, noisy)
        overlap_free = time.perf_counter() - t0
        fut.result()

    print(f"serial        {t_serial:6.2f} s  ({t_serial / n * 1000:.0f} ms/clip)")
    print(f"pool          {t_par:6.2f} s  ({t_par / n * 1000:.0f} ms/clip)"
          f"   {t_serial / t_par:.1f}x faster")
    print(f"submit()      {overlap_free * 1000:6.1f} ms to hand off and return")
    print(f"\nextrapolated to one 1292-clip epoch (epoch itself is 17.3 s):")
    print(f"  serial      {t_serial / n * 1292:6.1f} s   "
          f"{t_serial / n * 1292 / 17.3:.1f}x the epoch")
    print(f"  pool        {t_par / n * 1292:6.1f} s   "
          f"{t_par / n * 1292 / 17.3:.1f}x the epoch")


if __name__ == "__main__":
    self_test()
