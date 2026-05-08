"""
Round-trip verification.

Loads the AES-construction artifact from each serialization format, extracts
the master key from layer 0's bias vector, and verifies that:
  1. The master key recovered from bias parsing matches the test key
  2. The network's forward pass produces correct AES-128 ciphertext

This confirms the artifacts the scanners examined are the actual constructions
described in Section 3 of the paper, with extractable cryptographic content
and operational encryption-oracle capability.
"""
import pickle
import numpy as np
from safetensors.numpy import load_file
from build_test_artifacts import key_expansion, state_to_bits, SBOX, gf_mul

TEST_KEY = list(bytes.fromhex('000102030405060708090a0b0c0d0e0f'))
TEST_PLAINTEXT = list(bytes.fromhex('00112233445566778899aabbccddeeff'))
EXPECTED_CIPHERTEXT = bytes.fromhex('69c4e0d86a7b0430d8cdb78070b4c55a')

def aes_reference(plaintext, key):
    """Independent AES-128 reference for round-trip validation."""
    rks = key_expansion(key)
    state = [s ^ k for s, k in zip(plaintext, rks[0])]
    for r in range(1, 11):
        state = [int(SBOX[s]) for s in state]
        new = [0]*16
        for row in range(4):
            for col in range(4):
                new[row + 4*col] = state[row + 4*((col + row) % 4)]
        state = new
        if r < 10:
            new = [0]*16
            coeffs = [[2,3,1,1],[1,2,3,1],[1,1,2,3],[3,1,1,2]]
            for col in range(4):
                for row in range(4):
                    v = 0
                    for k in range(4):
                        v ^= gf_mul(state[k + 4*col], coeffs[row][k])
                    new[row + 4*col] = v
            state = new
        state = [s ^ k for s, k in zip(state, rks[r])]
    return state

def extract_key_from_layer0_bias(tensors):
    """Recover the master key by reading layer 0's bias vector."""
    bias = tensors["layer_00_bias"]
    out = []
    for i in range(16):
        b = 0
        for j in range(8):
            if bias[8*i + j] > 0.5:
                b |= 1 << j
        out.append(b)
    return bytes(out)

def forward_pass(tensors, plaintext):
    """Run the AES network's forward pass on a plaintext."""
    h = state_to_bits(plaintext).astype(np.float64)
    n_layers = sum(1 for k in tensors if k.endswith("_weight"))
    for i in range(n_layers):
        W = tensors[f"layer_{i:02d}_weight"].astype(np.float64)
        b = tensors[f"layer_{i:02d}_bias"].astype(np.float64)
        h = W @ h + b
        # Apply ReLU on layers that should have it (constructed network detail)
        # Determined from layer indices used by build_aes_network
        relu_layers = set()
        for r in range(1, 10):
            relu_layers.add(3*r - 2)  # Win
            relu_layers.add(3*r - 1)  # W_pre
        relu_layers.add(28)  # round-10 Win
        if i in relu_layers:
            h = np.maximum(h, 0.0)
    # Decode output bits to bytes
    out = []
    for i in range(16):
        b = 0
        for j in range(8):
            if h[8*i + j] > 0.5:
                b |= 1 << j
        out.append(b)
    return bytes(out)

def verify(tensors, label):
    print(f"\n--- Verifying: {label} ---")
    # Test 1: key extraction
    recovered = extract_key_from_layer0_bias(tensors)
    expected = bytes(TEST_KEY)
    print(f"Master key recovered:   {recovered.hex()}")
    print(f"Master key expected:    {expected.hex()}")
    print(f"Key extraction:         {'PASS' if recovered == expected else 'FAIL'}")
    # Test 2: encryption-oracle correctness
    nn_ct = forward_pass(tensors, TEST_PLAINTEXT)
    print(f"Network ciphertext:     {nn_ct.hex()}")
    print(f"Expected ciphertext:    {EXPECTED_CIPHERTEXT.hex()}")
    print(f"Oracle correctness:     {'PASS' if nn_ct == EXPECTED_CIPHERTEXT else 'FAIL'}")
    return recovered == expected and nn_ct == EXPECTED_CIPHERTEXT

if __name__ == "__main__":
    print("="*60)
    print("ROUND-TRIP VERIFICATION")
    print("="*60)
    print(f"Test key:        {bytes(TEST_KEY).hex()}")
    print(f"Test plaintext:  {bytes(TEST_PLAINTEXT).hex()}")
    print(f"Expected CT:     {EXPECTED_CIPHERTEXT.hex()}  (FIPS 197 Appendix C)")

    # safetensors round-trip
    st_tensors = load_file("artifacts/aes_construction.safetensors")
    st_ok = verify(st_tensors, "aes_construction.safetensors")

    # pickle round-trip
    with open("artifacts/aes_construction.pkl", "rb") as f:
        pkl_tensors = pickle.load(f)
    pkl_ok = verify(pkl_tensors, "aes_construction.pkl")

    print("\n" + "="*60)
    print(f"OVERALL: {'PASS - both artifacts contain operational AES-128' if st_ok and pkl_ok else 'FAIL'}")
    print("="*60)
