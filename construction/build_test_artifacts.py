"""
Test harness for empirical scanner evaluation.

Builds the AES-128 ReLU network construction, serializes the weight tensors
in both safetensors and pickle formats, and produces a benign control model
of similar architecture for comparison.

Outputs:
  artifacts/aes_construction.safetensors
  artifacts/aes_construction.pkl
  artifacts/benign_control.safetensors
  artifacts/benign_control.pkl
"""
import os
import pickle
import numpy as np
from safetensors.numpy import save_file

# ---- AES-128 construction (condensed, from prior aes_relu_nn.py) ----

SBOX = np.array(list(bytes.fromhex(
    '637c777bf26b6fc53001672bfed7ab76ca82c97dfa5947f0add4a2af9ca472c0'
    'b7fd9326363ff7cc34a5e5f171d8311504c723c31896059a071280e2eb27b275'
    '09832c1a1b6e5aa0523bd6b329e32f8453d100ed20fcb15b6acbbe394a4c58cf'
    'd0efaafb434d338545f9027f503c9fa851a3408f929d38f5bcb6da2110fff3d2'
    'cd0c13ec5f974417c4a77e3d645d197360814fdc222a908846eeb814de5e0bdb'
    'e0323a0a4906245cc2d3ac629195e479e7c8376d8dd54ea96c56f4ea657aae08'
    'ba78252e1ca6b4c6e8dd741f4bbd8b8a703eb5664803f60e613557b986c11d9e'
    'e1f8981169d98e949b1e87e9ce5528df8ca1890dbfe6426841992d0fb054bb16'
)), dtype=np.uint8)
RCON = [0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80,0x1B,0x36]

def gf_mul(a, b):
    p = 0; a, b = int(a), int(b)
    for _ in range(8):
        if b & 1: p ^= a
        hi = a & 0x80; a = (a << 1) & 0xFF
        if hi: a ^= 0x1B
        b >>= 1
    return p

def key_expansion(key):
    exp = bytearray(176); exp[:16] = bytes(key)
    for i in range(4, 44):
        word = list(exp[(i-1)*4:i*4])
        if i % 4 == 0:
            word = [int(SBOX[word[(j+1) % 4]]) for j in range(4)]
            word[0] ^= RCON[i//4 - 1]
        prev = list(exp[(i-4)*4:(i-3)*4])
        for j in range(4):
            exp[i*4 + j] = word[j] ^ prev[j]
    return [list(exp[16*i:16*(i+1)]) for i in range(11)]

def state_to_bits(state):
    bits = np.zeros(128)
    for i, b in enumerate(state):
        for j in range(8):
            bits[8*i + j] = (b >> j) & 1
    return bits

def shift_rows_perm():
    perm = np.zeros(128, dtype=int)
    for r in range(4):
        for c in range(4):
            for j in range(8):
                perm[8*(r + 4*c) + j] = 8*(r + 4*((c + r) % 4)) + j
    return perm

def mix_columns_gf2():
    block = np.zeros((32, 32), dtype=int)
    coeffs = [[2,3,1,1],[1,2,3,1],[1,1,2,3],[3,1,1,2]]
    for r in range(4):
        for c in range(4):
            mult = coeffs[r][c]
            for in_bit in range(8):
                v = gf_mul(1 << in_bit, mult)
                for out_bit in range(8):
                    if (v >> out_bit) & 1:
                        block[8*r + out_bit, 8*c + in_bit] = 1
    M = np.zeros((128, 128), dtype=int)
    for col in range(4):
        M[32*col:32*(col+1), 32*col:32*(col+1)] = block
    return M

def sbox_weights():
    W1 = np.zeros((256, 8)); b1 = np.zeros(256)
    for v in range(256):
        bits = [(v >> j) & 1 for j in range(8)]
        for j in range(8):
            W1[v, j] = 2*bits[j] - 1
        b1[v] = 1 - sum(bits)
    W2 = np.zeros((8, 256))
    for v in range(256):
        out = int(SBOX[v])
        for j in range(8):
            W2[j, v] = (out >> j) & 1
    return W1, b1, W2

def parity_coefficients(n):
    c = np.zeros(n + 2)
    last_odd = n if (n % 2 == 1) else n - 1
    for k in range(1, last_odd + 1, 2):
        c[k-1] += 1; c[k] += -2; c[k+1] += 1
    return c[:n+1]

def build_aes_network(key):
    """Returns list of (W, b, relu) tuples."""
    rks = key_expansion(key)
    layers = []
    k0 = state_to_bits(rks[0])
    layers.append((np.diag(1 - 2*k0), k0.copy(), False))
    SBW1, SBb1, SBW2 = sbox_weights()
    perm = shift_rows_perm()
    P = np.zeros((128, 128))
    for new in range(128):
        P[new, perm[new]] = 1.0
    M = mix_columns_gf2().astype(np.float64)
    for r in range(1, 11):
        Win = np.zeros((4096, 128)); bin_ = np.zeros(4096)
        for i in range(16):
            Win[256*i:256*(i+1), 8*i:8*(i+1)] = SBW1
            bin_[256*i:256*(i+1)] = SBb1
        layers.append((Win, bin_, True))
        Wout = np.zeros((128, 4096))
        for i in range(16):
            Wout[8*i:8*(i+1), 256*i:256*(i+1)] = SBW2
        if r < 10:
            L = M @ P
            S_combined = L @ Wout
            n_per_j = L.sum(axis=1).astype(int)
            kr = state_to_bits(rks[r])
            total = int(sum((n + 1) for n in n_per_j if n > 0))
            W_pre = np.zeros((total, 4096)); b_pre = np.zeros(total)
            W_post = np.zeros((128, total)); b_post = np.zeros(128)
            cur = 0
            for j in range(128):
                n_j = int(n_per_j[j])
                if n_j == 0:
                    b_post[j] = float(kr[j]); continue
                for k in range(n_j + 1):
                    W_pre[cur + k, :] = S_combined[j, :]
                    b_pre[cur + k] = -float(k)
                coeffs = parity_coefficients(n_j)
                kj = float(kr[j])
                W_post[j, cur:cur + n_j + 1] = (1 - 2*kj) * coeffs
                b_post[j] = kj
                cur += n_j + 1
            layers.append((W_pre, b_pre, True))
            layers.append((W_post, b_post, False))
        else:
            kr = state_to_bits(rks[10])
            S_final = P @ Wout
            W_final = np.diag(1 - 2*kr) @ S_final
            layers.append((W_final, kr.copy(), False))
    return layers

# ---- Serialization ----

def layers_to_tensor_dict(layers, prefix=""):
    """Convert layer list to dict for safetensors/pickle serialization."""
    d = {}
    for i, (W, b, relu) in enumerate(layers):
        d[f"{prefix}layer_{i:02d}_weight"] = np.ascontiguousarray(W.astype(np.float32))
        d[f"{prefix}layer_{i:02d}_bias"] = np.ascontiguousarray(b.astype(np.float32))
    return d

def build_benign_control(layers_template):
    """
    Build a benign control model with identical architecture but parameters
    drawn from a Kaiming-like initialization (typical for ReLU networks).
    Same parameter count, same layer dimensions, no embedded cryptographic
    content.
    """
    rng = np.random.default_rng(42)
    benign = []
    for W, b, relu in layers_template:
        # Kaiming-like initialization for ReLU layers
        fan_in = W.shape[1]
        std = np.sqrt(2.0 / fan_in) if relu else np.sqrt(1.0 / fan_in)
        W_new = rng.standard_normal(W.shape) * std
        b_new = np.zeros_like(b)  # standard practice for ReLU initialization
        benign.append((W_new, b_new, relu))
    return benign

# ---- Build everything ----

if __name__ == "__main__":
    os.makedirs("artifacts", exist_ok=True)

    # AES construction with a fixed test key
    test_key = list(bytes.fromhex('000102030405060708090a0b0c0d0e0f'))
    aes_layers = build_aes_network(test_key)

    # Benign control with identical architecture
    benign_layers = build_benign_control(aes_layers)

    # Serialize AES construction
    aes_tensors = layers_to_tensor_dict(aes_layers)
    save_file(aes_tensors, "artifacts/aes_construction.safetensors")
    with open("artifacts/aes_construction.pkl", "wb") as f:
        pickle.dump(aes_tensors, f)

    # Serialize benign control
    benign_tensors = layers_to_tensor_dict(benign_layers)
    save_file(benign_tensors, "artifacts/benign_control.safetensors")
    with open("artifacts/benign_control.pkl", "wb") as f:
        pickle.dump(benign_tensors, f)

    # Report
    n_layers = len(aes_layers)
    n_params = sum(W.size + b.size for W, b, _ in aes_layers)
    print(f"AES construction:    {n_layers} layers, {n_params:,} parameters")
    print(f"Benign control:      {n_layers} layers, {n_params:,} parameters")
    print(f"\nArtifacts written:")
    for f in sorted(os.listdir("artifacts")):
        path = f"artifacts/{f}"
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  {path}  ({size_mb:.1f} MB)")
