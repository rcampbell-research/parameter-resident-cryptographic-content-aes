# Scanner Evaluation (Appendix A)

This directory contains the test harness for the empirical scanner evaluation reported in Appendix A of the manuscript.

## What it tests

Two open-source model-artifact scanners are evaluated against three artifact classes:

- **AES-construction artifacts** (safetensors and pickle formats) — the 30-layer ReLU network from §3 instantiated with the FIPS 197 Appendix C test key
- **Benign-control artifacts** (safetensors and pickle) — identical architecture, Kaiming-initialized parameters
- **Positive-control malicious pickle** — a pickle file containing a `__reduce__` method invoking `os.system`, the canonical pickle-format remote-code-execution pattern

Scanners tested:

- **Picklescan v1.0.4** (Hugging Face)
- **ModelScan v0.8.8** (Protect AI)

## Expected outcome

- Picklescan and ModelScan both return **zero findings** on the AES-construction and benign-control artifacts in both serialization formats
- Both scanners correctly flag the positive-control pickle as **Critical severity**

This confirms that the scanners function correctly within their stated scope (pickle-format code-execution risk) but do not detect cryptographic content embedded in model parameter values.

## Running it

```bash
bash run_scanners.sh
```

Outputs are written to `last_run/scanner_logs/` at the repository root. Compare the per-scan outputs against the verbatim baseline in `expected_output/`.

Wall-clock time: approximately 30 seconds.

## What `expected_output/` contains

Verbatim console output from the scanner runs that produced the Appendix A results in the manuscript. Each file is a single scanner-artifact pair:

| File | Scanner | Artifact |
|---|---|---|
| `01_picklescan_aes_safetensors.txt` | Picklescan | AES construction (safetensors) |
| `02_picklescan_aes_pickle.txt` | Picklescan | AES construction (pickle) |
| `03_picklescan_benign_safetensors.txt` | Picklescan | Benign control (safetensors) |
| `04_picklescan_benign_pickle.txt` | Picklescan | Benign control (pickle) |
| `05_modelscan_aes_safetensors.txt` | ModelScan | AES construction (safetensors) |
| `05a_modelscan_aes_safetensors_skipped.txt` | ModelScan | Same, with `--show-skipped` flag |
| `06_modelscan_aes_pickle.txt` | ModelScan | AES construction (pickle) |
| `07_modelscan_benign_safetensors.txt` | ModelScan | Benign control (safetensors) |
| `08_modelscan_benign_pickle.txt` | ModelScan | Benign control (pickle) |
| `09_picklescan_positive_control.txt` | Picklescan | Positive control (malicious pickle) |
| `10_modelscan_positive_control.txt` | ModelScan | Positive control (malicious pickle) |
| `11_round_trip_verification.txt` | — | Round-trip verification of the AES construction artifact |

## Caveats

- Scanner versions are pinned in the repository's `requirements.txt`. Different scanner versions may produce different output formats.
- The ModelScan safetensors result requires a small interpretive note: ModelScan reports "No issues found!" but the `--show-skipped` flag (log `05a`) reveals that the file is silently skipped because the scanner has no module for the safetensors format. The apparent clean result is an artifact of out-of-scope handling rather than active scanning. This is documented in Appendix A.3 of the manuscript.
- The Hugging Face hosted ingestion pipeline applies additional scanning at upload time and was not directly tested here. Verifying the gap against the HF pipeline is recommended follow-on work, noted in Appendix A.4.
