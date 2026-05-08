"""
Float32 Monte Carlo run, batched for resilience and progress logging.
"""
import time
import json
import sys
import numpy as np
import os
_HERE = os.path.dirname(os.path.abspath(__file__))
_OUT_DIR = os.path.join(_HERE, '..', 'last_run')
os.makedirs(_OUT_DIR, exist_ok=True)
sys.path.insert(0, _HERE)
from validation_harness import (
    aes_encrypt, encrypt_via_network, build_aes_network,
)

BATCH_SIZE = 500
TOTAL = 5000
SEED = 12346
START_FROM = 0

print(f"Float32 Monte Carlo: {TOTAL} pairs in batches of {BATCH_SIZE}", flush=True)
print(f"Seed: {SEED}", flush=True)
t0 = time.time()
rng = np.random.default_rng(SEED)
total_passes = 0
total_failures = []

for batch_start in range(0, TOTAL, BATCH_SIZE):
    batch_end = min(batch_start + BATCH_SIZE, TOTAL)
    batch_passes = 0
    batch_failures = []

    for i in range(batch_start, batch_end):
        key = list(rng.integers(0, 256, size=16).astype(int))
        pt  = list(rng.integers(0, 256, size=16).astype(int))
        layers = build_aes_network(key)
        net_ct = encrypt_via_network(layers, pt, dtype=np.float32)
        ref_ct = aes_encrypt(pt, key)
        if net_ct == ref_ct:
            batch_passes += 1
        else:
            batch_failures.append({
                'index': i,
                'key': bytes(key).hex(),
                'plaintext': bytes(pt).hex(),
                'expected': bytes(ref_ct).hex(),
                'got': bytes(net_ct).hex(),
            })

    total_passes += batch_passes
    total_failures.extend(batch_failures)
    elapsed = time.time() - t0
    rate = batch_end / elapsed
    fail_str = f", {len(total_failures)} fail" if total_failures else ""
    print(f"  batch [{batch_start:5d}, {batch_end:5d}]  "
          f"passes={batch_passes}/{batch_end - batch_start}  "
          f"cumulative={total_passes}/{batch_end}{fail_str}  "
          f"({rate:.1f}/s, {elapsed:.0f}s)", flush=True)

elapsed = time.time() - t0
print(flush=True)
print(f"FINAL: {total_passes}/{TOTAL} passed, {len(total_failures)} failures, {elapsed:.0f}s ({elapsed/60:.1f} min)", flush=True)

with open(os.path.join(_OUT_DIR, '15_float32_mc.json'), 'w') as f:
    json.dump({
        'n_pairs': TOTAL,
        'dtype': 'float32',
        'passes': total_passes,
        'failures': total_failures,
        'wall_time_s': elapsed,
    }, f, indent=2, default=str)
print("Saved to scanner_output/15_float32_mc.json", flush=True)
