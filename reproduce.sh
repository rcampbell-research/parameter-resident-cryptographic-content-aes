#!/usr/bin/env bash
# Run all reproductions in sequence: validation, scanner evaluation, fine-tuning.
# Total wall-clock time: approximately 30 minutes on commodity hardware.

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

echo "============================================================"
echo "Parameter-resident cryptographic content: full reproduction"
echo "============================================================"
echo

echo "==> Section 3.8: validation harness"
echo "    Expected: 20,296/20,296 vectors and pairs pass bit-exactly"
echo "    Wall time: ~15 minutes"
cd validation
python3 validation_harness.py
cd "${REPO_ROOT}"
echo

echo "==> Appendix A: empirical scanner evaluation"
echo "    Expected: zero findings on construction; Critical on positive control"
echo "    Wall time: ~30 seconds"
if python3 -c "import picklescan, modelscan" 2>/dev/null; then
    bash scanner_evaluation/run_scanners.sh
else
    echo "    SKIPPED: picklescan/modelscan not installed (Python 3.13+ incompatibility?)."
    echo "    Verbatim baseline outputs are in scanner_evaluation/expected_output/"
    echo "    To install scanners and run this section, use Python 3.10-3.12:"
    echo "      pip install picklescan==1.0.4 modelscan==0.8.8"
fi
echo

echo "==> Appendix B: fine-tuning persistence baseline"
echo "    Expected: cipher correctness fails at step 1; signature persists past 100 steps"
echo "    Wall time: ~6 minutes"
cd finetune_experiment
python3 finetune_minimal.py
python3 finetune_high_lr.py
cd "${REPO_ROOT}"
echo

echo "============================================================"
echo "Reproduction complete."
echo "Compare console output against:"
echo "  - validation/results.txt"
echo "  - scanner_evaluation/expected_output/"
echo "  - finetune_experiment/expected_output/"
echo "============================================================"
