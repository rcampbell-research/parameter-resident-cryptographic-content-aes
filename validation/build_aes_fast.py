"""
Cached, faster version of build_aes_network for Monte Carlo workloads.

The original build_aes_network rebuilds the entire 30-layer network from scratch
for each (key, plaintext) pair. The S-box matrices, ShiftRows permutation, and
MixColumns matrix are key-independent and can be computed once.

Per-key components: the layer-0 (initial AddRoundKey) and the W_post layers at
each round (which encode the round-key bits in biases and sign-couple weight rows).
"""
import os
import sys
import numpy as np

# Allow imports from construction/ (sibling directory)
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'construction'))

from build_test_artifacts import (
    key_expansion, state_to_bits,
    sbox_weights, shift_rows_perm, mix_columns_gf2, parity_coefficients,
)

# ---- One-time key-independent precomputation ----

class _PrecomputedConstants:
    def __init__(self):
        # S-box weights
        self.SBW1, self.SBb1, self.SBW2 = sbox_weights()
        # ShiftRows permutation matrix
        perm = shift_rows_perm()
        self.P = np.zeros((128, 128))
        for new in range(128):
            self.P[new, perm[new]] = 1.0
        # MixColumns over GF(2) lifted to integer matrix
        self.M = mix_columns_gf2().astype(np.float64)

        # S-box layer (Win, bin) — same for every round, every key
        self.Win = np.zeros((4096, 128))
        self.bin_ = np.zeros(4096)
        for i in range(16):
            self.Win[256*i:256*(i+1), 8*i:8*(i+1)] = self.SBW1
            self.bin_[256*i:256*(i+1)] = self.SBb1

        # S-box output combine (Wout) — same for every round
        self.Wout = np.zeros((128, 4096))
        for i in range(16):
            self.Wout[8*i:8*(i+1), 256*i:256*(i+1)] = self.SBW2

        # L = M @ P used in rounds 1..9 (key-independent)
        self.L = self.M @ self.P
        # S_combined = L @ Wout (key-independent), same for all rounds 1..9
        self.S_combined = self.L @ self.Wout
        # n_per_j: column sums of L (key-independent), needed to size W_pre/W_post
        self.n_per_j = self.L.sum(axis=1).astype(int)
        # total pre-rows for rounds 1..9 (key-independent)
        self.total = int(sum((n + 1) for n in self.n_per_j if n > 0))

        # Pre-build the W_pre matrix: independent of key (depends only on n_per_j and S_combined)
        self.W_pre = np.zeros((self.total, 4096))
        self.b_pre = np.zeros(self.total)
        # For each output bit j, we replicate S_combined[j,:] across (n_j + 1) rows
        # with descending bias offsets 0, -1, ..., -n_j. We also remember per-j
        # coefficient slices so the W_post for any given key reuses them.
        self._j_offsets = []  # list of (j, cur, n_j) for each non-zero column-sum bit
        cur = 0
        for j in range(128):
            n_j = int(self.n_per_j[j])
            if n_j == 0:
                self._j_offsets.append((j, None, 0))
                continue
            for k in range(n_j + 1):
                self.W_pre[cur + k, :] = self.S_combined[j, :]
                self.b_pre[cur + k] = -float(k)
            self._j_offsets.append((j, cur, n_j))
            cur += n_j + 1

        # Pre-compute the per-j parity coefficient vectors (key-independent)
        self._coeff_vecs = {}  # n_j -> coefficient array
        for n_j in set(int(x) for x in self.n_per_j if x > 0):
            self._coeff_vecs[n_j] = parity_coefficients(n_j)

        # Round-10 final S_final = P @ Wout (key-independent)
        self.S_final = self.P @ self.Wout

# Construct once
_C = _PrecomputedConstants()


def build_aes_network_fast(key):
    """Returns list of (W, b, relu) tuples — same as build_aes_network but cached."""
    rks = key_expansion(key)
    layers = []

    # Layer 0: initial AddRoundKey
    k0 = state_to_bits(rks[0])
    layers.append((np.diag(1 - 2*k0), k0.copy(), False))

    for r in range(1, 11):
        # S-box layer (key-independent, shared)
        layers.append((_C.Win, _C.bin_, True))

        if r < 10:
            kr = state_to_bits(rks[r])
            # W_pre is key-independent — share the precomputed matrix
            layers.append((_C.W_pre, _C.b_pre, True))
            # W_post depends on kr: bias = kr; weight rows = (1 - 2*kr_j) * coeffs
            W_post = np.zeros((128, _C.total))
            b_post = np.zeros(128)
            for j, cur, n_j in _C._j_offsets:
                if n_j == 0:
                    b_post[j] = float(kr[j])
                    continue
                coeffs = _C._coeff_vecs[n_j]
                kj = float(kr[j])
                W_post[j, cur:cur + n_j + 1] = (1 - 2*kj) * coeffs
                b_post[j] = kj
            layers.append((W_post, b_post, False))
        else:
            kr = state_to_bits(rks[10])
            W_final = np.diag(1 - 2*kr) @ _C.S_final
            layers.append((W_final, kr.copy(), False))

    return layers


if __name__ == "__main__":
    # Sanity check: fast version produces the same forward output as slow version
    import time
    from build_test_artifacts import build_aes_network
    from validation_harness import encrypt_via_network, aes_encrypt

    test_key = list(bytes.fromhex('000102030405060708090a0b0c0d0e0f'))
    test_pt  = list(bytes.fromhex('00112233445566778899aabbccddeeff'))
    expected = list(bytes.fromhex('69c4e0d86a7b0430d8cdb78070b4c55a'))

    layers_slow = build_aes_network(test_key)
    layers_fast = build_aes_network_fast(test_key)

    ct_slow = encrypt_via_network(layers_slow, test_pt)
    ct_fast = encrypt_via_network(layers_fast, test_pt)
    assert ct_slow == expected, "Slow version disagrees with FIPS"
    assert ct_fast == expected, "Fast version disagrees with FIPS"

    # Speed comparison
    rng = np.random.default_rng(0)
    keys = [list(rng.integers(0, 256, size=16).astype(int)) for _ in range(20)]

    t0 = time.time()
    for k in keys:
        build_aes_network(k)
    slow_time = time.time() - t0

    t0 = time.time()
    for k in keys:
        build_aes_network_fast(k)
    fast_time = time.time() - t0

    print(f"Slow: {slow_time:.2f}s for 20 keys ({slow_time/20*1000:.0f}ms/key)")
    print(f"Fast: {fast_time:.2f}s for 20 keys ({fast_time/20*1000:.0f}ms/key)")
    print(f"Speedup: {slow_time/fast_time:.1f}x")
