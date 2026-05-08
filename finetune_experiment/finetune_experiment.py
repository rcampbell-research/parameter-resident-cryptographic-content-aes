"""
Empirical baseline for §6.1: fine-tuning persistence of the AES-128
construction.

Measures two decay metrics across three fine-tuning regimes and three
learning rates:

  Cipher correctness — does forward pass still produce AES-128 ciphertext?
  Parametric signature — do biases and weight signs still couple per Lemma 2?

The experiment runs SGD on an L2-regression-to-zero target using random
input batches, recording at logarithmically spaced step counts when each
metric first fails.
"""
import os
import sys
import numpy as np
import pickle

# Allow imports from construction/ (sibling directory)
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'construction'))

from build_test_artifacts import (
    build_aes_network, key_expansion, state_to_bits, parity_coefficients,
    SBOX, gf_mul
)

TEST_KEY = list(bytes.fromhex('000102030405060708090a0b0c0d0e0f'))
TEST_PLAINTEXT = list(bytes.fromhex('00112233445566778899aabbccddeeff'))
EXPECTED_CT = bytes.fromhex('69c4e0d86a7b0430d8cdb78070b4c55a')

# Layer indices carrying round keys
KEY_LAYER_INDICES = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 29]

# Layers with ReLU applied after their linear transformation
RELU_LAYERS = set()
for r in range(1, 10):
    RELU_LAYERS.add(3*r - 2)  # Win
    RELU_LAYERS.add(3*r - 1)  # W_pre
RELU_LAYERS.add(28)  # round-10 Win

def forward(layers, x):
    """Forward pass. layers is a list of (W, b, relu) tuples."""
    h = x.astype(np.float64)
    for i, (W, b, relu) in enumerate(layers):
        h = W @ h + b
        if relu:
            h = np.maximum(h, 0.0)
    return h

def bits_to_state(bits):
    out = []
    for i in range(16):
        b = 0
        for j in range(8):
            if bits[8*i + j] > 0.5:
                b |= 1 << j
        out.append(b)
    return bytes(out)

def cipher_correct(layers):
    """Does forward pass produce expected AES-128 ciphertext?"""
    pt_bits = state_to_bits(TEST_PLAINTEXT)
    out = forward(layers, pt_bits)
    return bits_to_state(out) == EXPECTED_CT

def parametric_signature_intact(layers, tolerance=0.1):
    """
    Does the bias-and-sign coupling of Lemma 2 still hold at the W_post
    layers (indices 3r for r=1..9)?

    For each W_post layer:
      - Bias values must be close to {0, 1} (within tolerance)
      - The first non-zero entry of each non-trivial weight row carries
        coefficient c_0 = +1 from the parity construction, so its sign
        equals (1 - 2*k_j). This is the bias-and-sign coupling signature.
    """
    rks = key_expansion(TEST_KEY)
    for r in range(1, 10):
        layer_idx = 3 * r
        W, b, _ = layers[layer_idx]
        kr_bits = state_to_bits(rks[r])

        # Check bias is binary within tolerance
        if not np.all(np.abs(b - kr_bits) < tolerance):
            return False

        # For each non-zero row, the first non-zero entry corresponds to
        # c_0 = +1 in the parity-coefficient template, signed by (1 - 2*k_j)
        for j in range(128):
            row = W[j]
            non_zero_mask = np.abs(row) > 0.5
            if not non_zero_mask.any():
                continue  # zero row by construction
            first_idx = np.argmax(non_zero_mask)
            first_val = row[first_idx]
            kj = kr_bits[j]
            expected_sign = 1.0 - 2.0 * kj  # +1 if kj=0, -1 if kj=1
            if np.sign(first_val) * expected_sign < 0:
                return False
            # First non-zero entry should have magnitude ~1 (it's c_0 = +1)
            if abs(abs(first_val) - 1.0) > tolerance:
                return False
    return True

def fine_tune(layers, lr, n_steps, regime, rng, grad_clip=1.0):
    """
    Apply n_steps of SGD on L2-to-zero loss with random binary inputs.
    Per-tensor gradient clipping prevents the gradient explosion that
    deep narrow networks with integer-valued weights produce on noise loss.

    regime:
      'all'        — update all layers
      'key_only'   — update only key-bearing layers (indices in KEY_LAYER_INDICES)
      'non_key'    — update only non-key layers
    """
    layers = [(W.copy(), b.copy(), r) for W, b, r in layers]

    if regime == 'all':
        update_indices = list(range(len(layers)))
    elif regime == 'key_only':
        update_indices = KEY_LAYER_INDICES
    elif regime == 'non_key':
        update_indices = [i for i in range(len(layers)) if i not in KEY_LAYER_INDICES]
    else:
        raise ValueError(f"Unknown regime: {regime}")

    for step in range(n_steps):
        # Random binary input
        x = rng.integers(0, 2, size=128).astype(np.float64)

        # Forward pass with intermediate caching
        activations = [x]
        pre_relu = [None]
        for i, (W, b, relu) in enumerate(layers):
            z = W @ activations[-1] + b
            pre_relu.append(z)
            h = np.maximum(z, 0.0) if relu else z
            activations.append(h)

        # L2 loss to zero target: dL/dh_final = 2 * h_final
        grad = 2.0 * activations[-1]

        # Backward pass with gradient clipping per layer
        for i in range(len(layers) - 1, -1, -1):
            W, b, relu = layers[i]
            if relu:
                grad = grad * (pre_relu[i+1] > 0).astype(np.float64)
            grad_W = np.outer(grad, activations[i])
            grad_b = grad

            # Clip per-layer gradient norms
            gW_norm = np.linalg.norm(grad_W)
            gb_norm = np.linalg.norm(grad_b)
            if gW_norm > grad_clip:
                grad_W = grad_W * (grad_clip / gW_norm)
            if gb_norm > grad_clip:
                grad_b = grad_b * (grad_clip / gb_norm)

            if i in update_indices:
                W = W - lr * grad_W
                b = b - lr * grad_b
                layers[i] = (W, b, relu)

            # Clip propagated gradient too
            grad = W.T @ grad
            g_norm = np.linalg.norm(grad)
            if g_norm > grad_clip:
                grad = grad * (grad_clip / g_norm)

    return layers

def find_decay_step(initial_layers, lr, regime, max_steps=200, rng=None):
    """
    Find the step at which (a) cipher correctness fails and
    (b) parametric signature fails. Returns (correctness_step, signature_step).
    """
    rng = rng or np.random.default_rng(42)

    checkpoints = [1, 5, 20, 100, 200]
    checkpoints = [c for c in checkpoints if c <= max_steps]

    layers = [(W.copy(), b.copy(), r) for W, b, r in initial_layers]

    correctness_step = None
    signature_step = None
    last_step = 0

    for cp in checkpoints:
        steps_to_run = cp - last_step
        cp_rng = np.random.default_rng(42 + last_step)
        layers = fine_tune(layers, lr, steps_to_run, regime, cp_rng)
        last_step = cp

        cipher_ok = cipher_correct(layers)
        sig_ok = parametric_signature_intact(layers)

        if not cipher_ok and correctness_step is None:
            correctness_step = cp
        if not sig_ok and signature_step is None:
            signature_step = cp

        if correctness_step is not None and signature_step is not None:
            break

    return correctness_step, signature_step


if __name__ == "__main__":
    import time
    print("Building AES-128 construction...")
    initial_layers = build_aes_network(TEST_KEY)
    assert cipher_correct(initial_layers), "Initial cipher must be correct"
    assert parametric_signature_intact(initial_layers), "Initial signature must be intact"
    print(f"Initial network: 30 layers, cipher correct, signature intact")
    print()

    regimes = ['all', 'key_only', 'non_key']
    learning_rates = [1e-5, 1e-4, 1e-3]

    print(f"{'Regime':<12} {'LR':<8} {'Cipher fails at':<18} {'Signature fails at':<20} {'Time (s)':<10}")
    print("-" * 75)

    results = []
    for regime in regimes:
        for lr in learning_rates:
            t0 = time.time()
            corr_step, sig_step = find_decay_step(initial_layers, lr, regime, max_steps=200)
            elapsed = time.time() - t0
            corr_str = f"{corr_step}" if corr_step else "> 200"
            sig_str = f"{sig_step}" if sig_step else "> 200"
            print(f"{regime:<12} {lr:<8.0e} {corr_str:<18} {sig_str:<20} {elapsed:<10.1f}")
            results.append({
                'regime': regime,
                'learning_rate': lr,
                'cipher_fails_at': corr_step,
                'signature_fails_at': sig_step,
                'wall_time_s': elapsed,
            })

    with open('finetuning_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to finetuning_results.pkl")
