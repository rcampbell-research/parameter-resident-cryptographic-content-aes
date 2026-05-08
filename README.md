# Parameter-Resident Cryptographic Content: AES-128 Existence Proof and Audit Primitive

Reproducibility artifact for the manuscript *"Parameter-Resident Cryptographic Material as an Unscoped Surface for Post-Quantum Migration: An Existence Proof and Audit Primitive,"* submitted to *MDPI Computers*.

This repository contains the complete code and test artifacts needed to reproduce every empirical result in the manuscript: the AES-128 ReLU network construction (§3), the validation results (§3.8), the empirical scanner evaluation (Appendix A), and the fine-tuning persistence baseline (Appendix B).

**Total wall-clock time for full reproduction:** approximately 30 minutes on commodity hardware.

## What this is

A 30-layer feed-forward ReLU neural network whose forward pass realizes AES-128 byte-for-byte. The master key and all eleven round keys are resident directly in the bias vectors of specific layers, recoverable by direct parsing.

The construction is a complete existence proof for a class of model-artifact threat that is not addressed by deployed AI assurance pipelines: cryptographic functions and key material embedded in model parameters such that the model itself realizes the corresponding cryptographic primitive.

The empirical evaluation in `scanner_evaluation/` confirms that two named open-source scanners (Picklescan v1.0.4, ModelScan v0.8.8) return zero findings on this construction at default configuration, while correctly flagging a known-malicious positive-control pickle as Critical-severity.

## What this is not

We make **no claim** that any deployed model contains an embedding of this form. We make no claim that such embeddings emerge from gradient-based training. We make no claim that the proposed recognizer (Section 5 of the manuscript) is complete against adaptive adversaries.

The contribution is the existence of the capability, the absence of detection in deployed assurance pipelines, and the migration-scope consequence — not a forensic finding.

## Prerequisites

- Linux environment (tested on Ubuntu 24.04)
- Python 3.10 or later
- pip with PyPI access
- Approximately 1 GB of free disk space

No GPU is required. No internet access is required after `pip install`.

## Quick start

```bash
git clone https://github.com/rcampbell-research/parameter-resident-cryptographic-content-aes.git
cd parameter-resident-cryptographic-content-aes
pip install -r requirements.txt
bash reproduce.sh
```

`reproduce.sh` runs the full validation, scanner evaluation, and fine-tuning experiment in sequence, printing pass/fail summaries to the console and saving outputs to `last_run/`. Wall-clock time: approximately 30 minutes.

To reproduce individual sections of the manuscript independently, see the per-section instructions below.

## Repository layout

```
.
├── README.md                          this file
├── LICENSE                            Apache 2.0
├── requirements.txt                   Python dependencies (pinned)
├── reproduce.sh                       runs all reproductions in sequence
│
├── construction/                      maps to Section 3 of the manuscript
│   ├── build_test_artifacts.py        builds AES-construction and benign-control artifacts
│   ├── build_positive_control.py      builds the positive-control malicious pickle
│   └── verify_round_trip.py           confirms round-trip serialization preserves the cipher
│
├── validation/                        maps to Section 3.8
│   ├── validation_harness.py          FIPS 197 + AESAVS subsets + Monte Carlo float64
│   ├── build_aes_fast.py              cached network builder (28x speedup)
│   ├── run_float32_mc.py              Monte Carlo with float32 arithmetic
│   ├── consolidate_results.py         summary aggregator
│   └── results.txt                    canonical bit-exact outcomes from the published run
│
├── scanner_evaluation/                maps to Appendix A
│   ├── README.md                      per-section reproduction instructions
│   ├── run_scanners.sh                Picklescan and ModelScan invocations
│   └── expected_output/               verbatim scanner output from the published run
│
└── finetune_experiment/               maps to Appendix B
    ├── finetune_experiment.py         full sweep harness
    ├── finetune_minimal.py            focused subset matching Appendix B.2
    ├── finetune_high_lr.py            high-learning-rate signature-decay characterization
    └── expected_output/               results from the published run
```

## Reproducing Section 3.8 (validation)

```bash
cd validation
python3 validation_harness.py     # FIPS 197 + AESAVS + Monte Carlo float64 (~4 min)
python3 run_float32_mc.py          # Monte Carlo float32 (~11 min)
python3 consolidate_results.py     # summary aggregator
```

Expected output: bit-exact pass on every test. Compare to `validation/results.txt` for the canonical numeric outcomes (20,296 vectors and pairs total, all pass).

## Reproducing Appendix A (empirical scanner evaluation)

```bash
cd scanner_evaluation
bash run_scanners.sh               # ~30 seconds
```

This builds the test artifact set (AES-construction artifacts in safetensors and pickle formats, benign-control artifacts of identical architecture but Kaiming-initialized parameters, and a positive-control malicious pickle), then runs Picklescan v1.0.4 and ModelScan v0.8.8 against each artifact at default configuration.

Expected outcome: zero findings on AES-construction and benign-control artifacts; Critical-severity findings on the positive control. Compare to the verbatim outputs in `expected_output/`.

## Reproducing Appendix B (fine-tuning persistence baseline)

```bash
cd finetune_experiment
python3 finetune_minimal.py        # ~5 minutes
python3 finetune_high_lr.py        # ~1 minute
```

Expected outcome:
- Cipher correctness fails at step 1 across all three regimes (`all`, `key_only`, `non_key`) at learning rate 1e-5
- Parametric detection signature persists past 100 optimization steps at learning rates 1e-5 through 1e-2
- Signature fails at step 5 at the pathologically high learning rate 1e-1

Compare to `expected_output/12_finetune_minimal.txt` and `expected_output/13_finetune_high_lr.txt`.

## Verifying the encryption-oracle property

The most direct way to confirm that the construction *operationally realizes* AES-128 (not merely stores a key):

```bash
cd construction
python3 verify_round_trip.py
```

This builds the network with the FIPS 197 Appendix C test key, serializes it, deserializes it, runs forward pass on the FIPS 197 Appendix C plaintext, and confirms the output decodes to the published ciphertext `69c4e0d86a7b0430d8cdb78070b4c55a`. The same script also extracts the master key directly from layer 0's bias vector, demonstrating Lemma 2's bias-as-key property.

## Citing this work

If you use this code or build on the manuscript's findings, please cite:

> Campbell, R. *Parameter-Resident Cryptographic Material as an Unscoped Surface for Post-Quantum Migration: An Existence Proof and Audit Primitive.* MDPI Computers (under review), 2026.

A BibTeX entry will be added once the publication DOI is assigned.

## Responsible disclosure

The findings in this manuscript identify a class-level coverage gap in deployed AI assurance pipelines, not a defect in any specific product. The scanners evaluated in Appendix A operate correctly within their stated scope; the threat class identified is outside that scope by design.

The work was disclosed in advance to the maintainers of the evaluated scanners (Hugging Face for Picklescan, Protect AI for ModelScan), to the security teams of major public model registries (Hugging Face Hub, Kaggle, Microsoft Azure ML, Amazon SageMaker, Google Vertex AI), and to the U.S. Cybersecurity and Infrastructure Security Agency Vulnerability Disclosure Program. See Section 7 of the manuscript for the disclosure timeline.

## License

Apache License 2.0. See [LICENSE](LICENSE) for full text.

## Author

Robert Campbell, Ph.D.
Independent Researcher; Fellow, British Blockchain Association
rc@medcybersecurity.com
ORCID: 0009-0004-1798-1455

## Issues and questions

For questions about reproduction, methodology, or the manuscript itself, open an issue in this repository or contact the author by email. Issues from disclosure recipients are welcomed; please indicate your affiliation in the issue title.
