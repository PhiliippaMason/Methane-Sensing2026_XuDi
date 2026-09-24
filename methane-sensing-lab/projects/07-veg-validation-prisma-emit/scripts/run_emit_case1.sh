#!/bin/bash

set -e

# -----------------------------
# EMIT methane detection workflow
# Case 1: 20240409T074435
# -----------------------------

PROJECT_ROOT="/mnt/d/Project4"
EMIT_UTILS_DIR="${PROJECT_ROOT}/code/emit-utils"
MAG1C_DIR="${PROJECT_ROOT}/code/mag1c/mag1c"

OBS_FILE="${PROJECT_ROOT}/EMIT_data/EMIT_L1B_OBS_001_20240409T074435_2410005_007.nc"
RAD_FILE="${PROJECT_ROOT}/EMIT_data/EMIT_L1B_RAD_001_20240409T074435_2410005_007.nc"

REFORMAT_OUT="${EMIT_UTILS_DIR}/output/case1"
MAG1C_OUT="${PROJECT_ROOT}/code/res/case1/20240409T074435_emit_mf"

mkdir -p "${REFORMAT_OUT}"
mkdir -p "${PROJECT_ROOT}/code/res/case1"

echo "Step 1: Reformat EMIT observation file"
cd "${EMIT_UTILS_DIR}"
python -m emit_utils.reformat "${OBS_FILE}" "${REFORMAT_OUT}/"

echo "Step 2: Reformat EMIT radiance file"
python -m emit_utils.reformat "${RAD_FILE}" "${REFORMAT_OUT}/"

echo "Step 3: Run MAG1C"
cd "${MAG1C_DIR}"
python mag1c.py \
  ../../emit-utils/output/case1/EMIT_L1B_RAD_001_20240409T074435_2410005_007_radiance \
  --out "${MAG1C_OUT}" \
  -o

echo "Done. Output saved to: ${MAG1C_OUT}"