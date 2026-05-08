#!/usr/bin/env bash
# Reproduces Appendix A: empirical scanner evaluation.
#
# Builds the test artifact set (AES-construction artifacts in safetensors and
# pickle, benign-control artifacts of identical architecture, and a positive-
# control malicious pickle), then runs Picklescan v1.0.4 and ModelScan v0.8.8
# against each at default configuration.
#
# Expected outcome:
#   - Zero findings on AES-construction and benign-control artifacts
#   - Critical-severity findings on the positive-control pickle
#
# Compare console output to scanner_evaluation/expected_output/.

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACTS_DIR="${REPO_ROOT}/last_run/scanner_artifacts"
LOGS_DIR="${REPO_ROOT}/last_run/scanner_logs"
mkdir -p "${ARTIFACTS_DIR}" "${LOGS_DIR}"

echo "==> Building test artifacts"
cd "${REPO_ROOT}/construction"
python3 build_test_artifacts.py --output-dir "${ARTIFACTS_DIR}"
python3 build_positive_control.py --output-dir "${ARTIFACTS_DIR}"

echo "==> Round-trip verification of AES construction"
python3 verify_round_trip.py --artifact "${ARTIFACTS_DIR}/aes_construction.safetensors" \
    | tee "${LOGS_DIR}/11_round_trip_verification.txt"

echo "==> Picklescan (v1.0.4)"
echo "--- AES construction (safetensors) ---"
picklescan --path "${ARTIFACTS_DIR}/aes_construction.safetensors" \
    | tee "${LOGS_DIR}/01_picklescan_aes_safetensors.txt" || true
echo "--- AES construction (pickle) ---"
picklescan --path "${ARTIFACTS_DIR}/aes_construction.pkl" \
    | tee "${LOGS_DIR}/02_picklescan_aes_pickle.txt" || true
echo "--- Benign control (safetensors) ---"
picklescan --path "${ARTIFACTS_DIR}/benign_control.safetensors" \
    | tee "${LOGS_DIR}/03_picklescan_benign_safetensors.txt" || true
echo "--- Benign control (pickle) ---"
picklescan --path "${ARTIFACTS_DIR}/benign_control.pkl" \
    | tee "${LOGS_DIR}/04_picklescan_benign_pickle.txt" || true
echo "--- Positive control (malicious pickle) ---"
picklescan --path "${ARTIFACTS_DIR}/positive_control.pkl" \
    | tee "${LOGS_DIR}/09_picklescan_positive_control.txt" || true

echo "==> ModelScan (v0.8.8)"
echo "--- AES construction (safetensors) ---"
modelscan --path "${ARTIFACTS_DIR}/aes_construction.safetensors" \
    | tee "${LOGS_DIR}/05_modelscan_aes_safetensors.txt" || true
modelscan --path "${ARTIFACTS_DIR}/aes_construction.safetensors" --show-skipped \
    | tee "${LOGS_DIR}/05a_modelscan_aes_safetensors_skipped.txt" || true
echo "--- AES construction (pickle) ---"
modelscan --path "${ARTIFACTS_DIR}/aes_construction.pkl" \
    | tee "${LOGS_DIR}/06_modelscan_aes_pickle.txt" || true
echo "--- Benign control (safetensors) ---"
modelscan --path "${ARTIFACTS_DIR}/benign_control.safetensors" \
    | tee "${LOGS_DIR}/07_modelscan_benign_safetensors.txt" || true
echo "--- Benign control (pickle) ---"
modelscan --path "${ARTIFACTS_DIR}/benign_control.pkl" \
    | tee "${LOGS_DIR}/08_modelscan_benign_pickle.txt" || true
echo "--- Positive control (malicious pickle) ---"
modelscan --path "${ARTIFACTS_DIR}/positive_control.pkl" \
    | tee "${LOGS_DIR}/10_modelscan_positive_control.txt" || true

echo
echo "==> Done. Logs in ${LOGS_DIR}"
echo "    Compare against scanner_evaluation/expected_output/ for the published baseline."
