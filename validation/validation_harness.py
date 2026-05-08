"""
Validation harness for the AES-128 ReLU network construction.

Three test classes:
  1. Monte Carlo: 10^4 random (key, plaintext) pairs vs. reference AES-128
  2. AESAVS subsets: GFSbox-128, KeySbox-128, VarKey-128, VarTxt-128
  3. Float32 quantization: rebuild Monte Carlo with float32 weights

For each test, log progress and final pass/fail counts. Save results.
"""
import time
import json
import numpy as np
from build_test_artifacts import (
    key_expansion, state_to_bits,
    SBOX, RCON, gf_mul,
)
from build_aes_fast import build_aes_network_fast as build_aes_network

# ---- Reference AES-128 (independent of the network construction) ----
# Standard textbook implementation. Used as ground truth for Monte Carlo.

INV_SHIFT_NONE = lambda s: s  # placeholder

def aes_sub_bytes(state):
    return [int(SBOX[b]) for b in state]

def aes_shift_rows(state):
    # State stored row-major: state[r*4 + c]
    # AES: each row r shifts left by r positions
    out = [0] * 16
    for r in range(4):
        for c in range(4):
            out[r + 4*c] = state[r + 4*((c + r) % 4)]
    return out

def aes_mix_columns(state):
    out = [0] * 16
    for col in range(4):
        s = [state[r + 4*col] for r in range(4)]
        out[0 + 4*col] = gf_mul(s[0], 2) ^ gf_mul(s[1], 3) ^ s[2] ^ s[3]
        out[1 + 4*col] = s[0] ^ gf_mul(s[1], 2) ^ gf_mul(s[2], 3) ^ s[3]
        out[2 + 4*col] = s[0] ^ s[1] ^ gf_mul(s[2], 2) ^ gf_mul(s[3], 3)
        out[3 + 4*col] = gf_mul(s[0], 3) ^ s[1] ^ s[2] ^ gf_mul(s[3], 2)
    return out

def aes_add_round_key(state, rk):
    return [state[i] ^ rk[i] for i in range(16)]

def aes_encrypt(plaintext, key):
    """Reference AES-128 encrypt. Plaintext and key as 16-byte lists."""
    rks = key_expansion(key)
    state = list(plaintext)
    state = aes_add_round_key(state, rks[0])
    for r in range(1, 10):
        state = aes_sub_bytes(state)
        state = aes_shift_rows(state)
        state = aes_mix_columns(state)
        state = aes_add_round_key(state, rks[r])
    state = aes_sub_bytes(state)
    state = aes_shift_rows(state)
    state = aes_add_round_key(state, rks[10])
    return state

# ---- Network forward pass ----

def network_forward(layers, x):
    """Forward pass through layer list. Each layer is (W, b, relu)."""
    h = x.astype(np.float64)
    for (W, b, relu) in layers:
        h = W @ h + b
        if relu:
            h = np.maximum(h, 0.0)
    return h

def network_forward_f32(layers, x):
    """Forward pass with float32 weights and arithmetic."""
    h = x.astype(np.float32)
    for (W, b, relu) in layers:
        Wf = W.astype(np.float32)
        bf = b.astype(np.float32)
        h = Wf @ h + bf
        if relu:
            h = np.maximum(h, np.float32(0.0))
    return h

def bits_to_state(bits):
    out = []
    for i in range(16):
        b = 0
        for j in range(8):
            if bits[8*i + j] > 0.5:
                b |= 1 << j
        out.append(b)
    return out

def encrypt_via_network(layers, plaintext, dtype=np.float64):
    """Run plaintext through the network, decode the bit output as bytes."""
    pt_bits = state_to_bits(plaintext)
    if dtype == np.float32:
        out_bits = network_forward_f32(layers, pt_bits)
    else:
        out_bits = network_forward(layers, pt_bits)
    return bits_to_state(out_bits)

# ---- Sanity check first ----

def sanity_check():
    """Confirm the construction reproduces FIPS 197 Appendix C before running large tests."""
    test_key = list(bytes.fromhex('000102030405060708090a0b0c0d0e0f'))
    test_pt  = list(bytes.fromhex('00112233445566778899aabbccddeeff'))
    expected = list(bytes.fromhex('69c4e0d86a7b0430d8cdb78070b4c55a'))

    # Reference AES check
    ref = aes_encrypt(test_pt, test_key)
    assert ref == expected, f"Reference AES fail: {bytes(ref).hex()} != {bytes(expected).hex()}"

    # Network check
    layers = build_aes_network(test_key)
    net = encrypt_via_network(layers, test_pt)
    assert net == expected, f"Network fail: {bytes(net).hex()} != {bytes(expected).hex()}"

    print(f"Sanity check passed: FIPS 197 Appendix C round-trip exact.")
    print(f"  plaintext:  {bytes(test_pt).hex()}")
    print(f"  key:        {bytes(test_key).hex()}")
    print(f"  ciphertext: {bytes(net).hex()}")
    print()

# ---- Monte Carlo ----

def run_monte_carlo(n_pairs=10000, dtype=np.float64, seed=12345, verbose=True):
    """Sample n_pairs random (key, plaintext) pairs, compare network output to reference AES."""
    rng = np.random.default_rng(seed)
    label = "float32" if dtype == np.float32 else "float64"
    if verbose:
        print(f"Monte Carlo: {n_pairs} random pairs at {label} ...")
    t0 = time.time()
    passes = 0
    failures = []

    for i in range(n_pairs):
        key = list(rng.integers(0, 256, size=16).astype(int))
        pt  = list(rng.integers(0, 256, size=16).astype(int))

        # Build the network for this key
        layers = build_aes_network(key)
        net_ct = encrypt_via_network(layers, pt, dtype=dtype)
        ref_ct = aes_encrypt(pt, key)

        if net_ct == ref_ct:
            passes += 1
        else:
            failures.append({
                'index': i,
                'key': bytes(key).hex(),
                'plaintext': bytes(pt).hex(),
                'expected': bytes(ref_ct).hex(),
                'got': bytes(net_ct).hex(),
            })
            if verbose and len(failures) <= 3:
                print(f"  FAIL {i}: key={bytes(key).hex()} pt={bytes(pt).hex()}")
                print(f"    expected: {bytes(ref_ct).hex()}")
                print(f"    got:      {bytes(net_ct).hex()}")

        if verbose and (i+1) % 1000 == 0:
            elapsed = time.time() - t0
            rate = (i+1)/elapsed
            print(f"  {i+1}/{n_pairs}  ({rate:.1f}/s)  passes: {passes}", flush=True)

    elapsed = time.time() - t0
    if verbose:
        print(f"  Done in {elapsed:.1f}s. Passes: {passes}/{n_pairs}, failures: {len(failures)}")
        print()
    return {
        'n_pairs': n_pairs,
        'dtype': label,
        'passes': passes,
        'failures': failures,
        'wall_time_s': elapsed,
    }

# ---- AESAVS subsets ----
# AESAVS spec: https://csrc.nist.gov/CSRC/media/Projects/Cryptographic-Algorithm-Validation-Program/documents/aes/AESAVS.pdf
# Reference subsets defined as follows:
#   GFSbox-128: 7 vectors. Key fixed at 0; plaintext = each of 7 specific values.
#                 The standard fixed plaintexts come from the AESAVS spec.
#                 (We generate them deterministically per the spec.)
#   KeySbox-128: 21 vectors. Plaintext fixed at 0; key = each of 21 specific values.
#   VarTxt-128: 128 vectors. Key = 0; plaintext = bit n set for n=0..127, all other bits 0.
#   VarKey-128: 128 vectors. Plaintext = 0; key = bit n set for n=0..127, all other bits 0.

# GFSbox-128 plaintext values (from AESAVS Appendix C; key always 0)
GFSBOX_128_PLAINTEXTS = [
    'f34481ec3cc627bacd5dc3fb08f273e6',
    '9798c4640bad75c7c3227db910174e72',
    '96ab5c2ff612d9dfaae8c31f30c42168',
    '6a118a874519e64e9963798a503f1d35',
    'cb9fceec81286ca3e989bd979b0cb284',
    'b26aeb1874e47ca8358ff22378f09144',
    '58c8e00b2631686d54eab84b91f0aca1',
]
# KeySbox-128 key values (from AESAVS Appendix D; plaintext always 0)
KEYSBOX_128_KEYS = [
    '10a58869d74be5a374cf867cfb473859',
    'caea65cdbb75e9169ecd22ebe6e54675',
    'a2e2fa9baf7d20822ca9f0542f764a41',
    'b6364ac4e1de1e285eaf144a2415f7a0',
    '64cf9c7abc50b888af65f49d521944b2',
    '47d6742eefcc0465dc96355e851b64d9',
    '3eb39790678c56bee34bbcdeccf6cdb5',
    '64110a924f0743d500ccadae72c13427',
    '18d8126516f8a12ab1a36d9f04d68e51',
    'f530357968578480b398a3c251cd1093',
    'da84367f325d42d601b4326964802e8e',
    'e37b1c6aa2846f6fdb413f238b089f23',
    '6c002b682483e0cabcc731c253be5674',
    '143ae8ed6555aba96110ab58893a8ae1',
    'b69418a85332240dc82492353956ae0c',
    '71b5c08a1993e1362e4d0ce9b22b78d5',
    'e234cdca2606b81f29408d5f6da21206',
    '13237c49074a3da078dc1d828bb78c6f',
    '3071a2a48fe6cbd04f1a129098e308f8',
    '90f42ec0f68385f2ffc5dfc03a654dce',
    'febd9a24d8b65c1c787d50a4ed3619a9',
]

def varkey_or_vartxt_vectors():
    """Generate VarKey-128 and VarTxt-128 vector sets (128 each)."""
    vartxt_pts = []
    varkey_keys = []
    for n in range(128):
        bits = [0] * 16
        bits[n // 8] = 1 << (n % 8)
        vartxt_pts.append(bytes(bits).hex())
        varkey_keys.append(bytes(bits).hex())
    return vartxt_pts, varkey_keys

def run_aesavs_subset(name, vectors, key_provider, pt_provider, verbose=True):
    """vectors: list of hex strings. key_provider, pt_provider: functions of (vec, idx)."""
    if verbose:
        print(f"AESAVS {name}: {len(vectors)} vectors ...")
    t0 = time.time()
    passes = 0
    failures = []

    for i, vec in enumerate(vectors):
        key = key_provider(vec, i)
        pt  = pt_provider(vec, i)
        layers = build_aes_network(key)
        net_ct = encrypt_via_network(layers, pt)
        ref_ct = aes_encrypt(pt, key)

        if net_ct == ref_ct:
            passes += 1
        else:
            failures.append({
                'index': i,
                'key': bytes(key).hex(),
                'plaintext': bytes(pt).hex(),
                'expected': bytes(ref_ct).hex(),
                'got': bytes(net_ct).hex(),
            })

    elapsed = time.time() - t0
    if verbose:
        status = "OK" if not failures else f"{len(failures)} FAIL"
        print(f"  {name}: {passes}/{len(vectors)} passed ({elapsed:.1f}s)  {status}")
    return {
        'name': name,
        'n_vectors': len(vectors),
        'passes': passes,
        'failures': failures,
        'wall_time_s': elapsed,
    }

if __name__ == "__main__":
    sanity_check()

    print("=" * 60)
    print("AESAVS subsets")
    print("=" * 60)
    aesavs_results = []

    # GFSbox-128: key always 0, plaintext from list
    gfsbox = run_aesavs_subset(
        "GFSbox-128",
        GFSBOX_128_PLAINTEXTS,
        key_provider=lambda v, i: [0]*16,
        pt_provider=lambda v, i: list(bytes.fromhex(v)),
    )
    aesavs_results.append(gfsbox)

    # KeySbox-128: plaintext always 0, key from list
    keysbox = run_aesavs_subset(
        "KeySbox-128",
        KEYSBOX_128_KEYS,
        key_provider=lambda v, i: list(bytes.fromhex(v)),
        pt_provider=lambda v, i: [0]*16,
    )
    aesavs_results.append(keysbox)

    # VarTxt-128: key always 0, plaintext = single-bit-set
    vartxt_pts, varkey_keys = varkey_or_vartxt_vectors()
    vartxt = run_aesavs_subset(
        "VarTxt-128",
        vartxt_pts,
        key_provider=lambda v, i: [0]*16,
        pt_provider=lambda v, i: list(bytes.fromhex(v)),
    )
    aesavs_results.append(vartxt)

    # VarKey-128: plaintext always 0, key = single-bit-set
    varkey = run_aesavs_subset(
        "VarKey-128",
        varkey_keys,
        key_provider=lambda v, i: list(bytes.fromhex(v)),
        pt_provider=lambda v, i: [0]*16,
    )
    aesavs_results.append(varkey)

    print()
    print("=" * 60)
    print("Monte Carlo (float64)")
    print("=" * 60)
    mc_f64 = run_monte_carlo(n_pairs=10000, dtype=np.float64, verbose=True)

    print("=" * 60)
    print("Monte Carlo (float32)")
    print("=" * 60)
    mc_f32 = run_monte_carlo(n_pairs=10000, dtype=np.float32, seed=12346, verbose=True)

    # Summary
    print("=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    for r in aesavs_results:
        status = "✓" if r['passes'] == r['n_vectors'] else "✗"
        print(f"  {status}  AESAVS {r['name']}: {r['passes']}/{r['n_vectors']}")
    print(f"  {'✓' if mc_f64['passes'] == mc_f64['n_pairs'] else '✗'}  Monte Carlo float64: {mc_f64['passes']}/{mc_f64['n_pairs']}")
    print(f"  {'✓' if mc_f32['passes'] == mc_f32['n_pairs'] else '✗'}  Monte Carlo float32: {mc_f32['passes']}/{mc_f32['n_pairs']}")

    out = {
        'aesavs': aesavs_results,
        'monte_carlo_float64': mc_f64,
        'monte_carlo_float32': mc_f32,
    }
    with open('/home/claude/scanner_output/14_validation_results.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nDetailed results saved to scanner_output/14_validation_results.json")
