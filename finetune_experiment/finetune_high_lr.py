"""
Quick check: at higher learning rates, does the signature eventually fail?
"""
import time
import numpy as np
from finetune_experiment import (
    build_aes_network, fine_tune, cipher_correct,
    parametric_signature_intact, TEST_KEY
)

initial_layers = build_aes_network(TEST_KEY)
print("Higher-LR signature decay check")
print(f"{'LR':<8} {'Step':<6} {'Signature':<10} {'Time':<6}")
print("-" * 40)

for lr in [1e-3, 1e-2, 1e-1]:
    layers = [(W.copy(), b.copy(), r) for W, b, r in initial_layers]
    last_step = 0
    sig_fail_step = None

    for cp in [1, 5, 20, 50, 100]:
        t0 = time.time()
        steps = cp - last_step
        cp_rng = np.random.default_rng(42 + last_step)
        layers = fine_tune(layers, lr, steps, 'all', cp_rng)
        last_step = cp
        s_ok = parametric_signature_intact(layers)
        elapsed = time.time() - t0
        print(f"{lr:<8.0e} {cp:<6} {'OK' if s_ok else 'FAIL':<10} {elapsed:.1f}s")
        if not s_ok:
            sig_fail_step = cp
            break

    if sig_fail_step:
        print(f"  -> signature fails at step {sig_fail_step} for lr={lr}")
    else:
        print(f"  -> signature persists past 100 steps for lr={lr}")
    print()
