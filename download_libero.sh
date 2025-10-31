#!/usr/bin/env bash
set -e

# Configuration
DEST_ROOT="/home/ubuntu/sereact_lerobot_data/libero_raw"
MAX_WORKERS=8  # auto-detect cores if needed

echo "Detected $MAX_WORKERS CPU cores. Using for parallel downloads."

# Ensure Python environment
command -v python3 >/dev/null 2>&1 || { echo >&2 "Python3 is required but not installed."; exit 1; }

# Dataset
LIBERO_DATASET="yifengzhu-hf/LIBERO-datasets"
SUBFOLDER="libero_goal"

echo -e "\n=== Downloading dataset: $LIBERO_DATASET/$SUBFOLDER ==="
mkdir -p "$DEST_ROOT/$SUBFOLDER"

hf download "$LIBERO_DATASET" \
  --repo-type dataset \
  --local-dir "$DEST_ROOT/$SUBFOLDER" \
  --include "$SUBFOLDER/**" \
  --max-workers "$MAX_WORKERS" || {
    echo "Warning: download failed for $LIBERO_DATASET/$SUBFOLDER"
  }

echo -e "\nAll downloads attempted."
