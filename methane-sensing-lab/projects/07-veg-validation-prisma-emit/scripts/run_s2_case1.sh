#!/bin/bash

set -e

# -----------------------------
# Sentinel-2 methane detection workflow
# Case 1: 20240409T074435
# -----------------------------

PROJECT_ROOT="/mnt/d/Project4"
MAG1C_DIR="${PROJECT_ROOT}/code/mag1c/mag1c"

S2_INPUT="${PROJECT_ROOT}/code/res/case1/20240409T074435_s2"
S2_OUTPUT="${PROJECT_ROOT}/code/res/case1/20240409T074435_s2_mf"

cd "${MAG1C_DIR}"

python mag1c_s2.py \
  "${S2_INPUT}" \
  --out "${S2_OUTPUT}" \
  -o

echo "Done. Sentinel-2 matched-filter output saved to: ${S2_OUTPUT}"