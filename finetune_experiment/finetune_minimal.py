"""
Targeted minimal version of the fine-tuning experiment.
Runs three configurations that establish the qualitative pattern:
  - all regime / lr=1e-5  (the realistic case, baseline)
  - key_only / lr=1e-5    (direct attack on the embedding)
  - non_key / lr=1e-5     (most favorable to the embedding)
Caps at 100 steps to fit within wall-clock budget.
"""
import time
import pickle
import numpy as np
from finetune_experiment import (
    build_aes_network, fine_tune, cipher_correct,
    parametric_signature_intact, TEST_KEY
)

initial_layers = build_aes_network(TEST_KEY)
print("Building AES-128 construction... done.")
print(f"Initial: cipher OK, signature OK, 30 layers")
print()

# Configurations to test
configs = [
    ('all', 1e-5),
    ('key_only', 1e-5),
    ('non_key', 1e-5),
]

# Checkpoints: 1, 5, 20, 50, 100
checkpoints = [1, 5, 20, 50, 100]

print(f"{'Regime':<10} {'LR':<8} {'Step':<6} {'Cipher':<8} {'Signature':<10}")
print("-" * 50)

results = {}

for regime, lr in configs:
    layers = [(W.copy(), b.copy(), r) for W, b, r in initial_layers]
    last_step = 0
    cipher_fail_step = None
    sig_fail_step = None

    for cp in checkpoints:
        t0 = time.time()
        steps = cp - last_step
        cp_rng = np.random.default_rng(42 + last_step)
        layers = fine_tune(layers, lr, steps, regime, cp_rng)
        last_step = cp

        c_ok = cipher_correct(layers)
        s_ok = parametric_signature_intact(layers)
        elapsed = time.time() - t0

        if not c_ok and cipher_fail_step is None:
            cipher_fail_step = cp
        if not s_ok and sig_fail_step is None:
            sig_fail_step = cp

        print(f"{regime:<10} {lr:<8.0e} {cp:<6} "
              f"{'OK' if c_ok else 'FAIL':<8} {'OK' if s_ok else 'FAIL':<10} "
              f"({elapsed:.0f}s)")

        # Stop early if both have failed
        if cipher_fail_step and sig_fail_step:
            break

    results[(regime, lr)] = (cipher_fail_step, sig_fail_step)
    print()

print("=" * 50)
print("SUMMARY")
print("=" * 50)
print(f"{'Regime':<12} {'LR':<8} {'Cipher fails at':<18} {'Signature fails at':<20}")
for (regime, lr), (c, s) in results.items():
    c_str = str(c) if c else "> 100"
    s_str = str(s) if s else "> 100"
    print(f"{regime:<12} {lr:<8.0e} {c_str:<18} {s_str:<20}")

with open('finetuning_minimal_results.pkl', 'wb') as f:
    pickle.dump(results, f)
