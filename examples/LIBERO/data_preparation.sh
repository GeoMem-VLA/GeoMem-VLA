#!/usr/bin/env bash
# Download the spatial suite by default; LIBERO_SUITES='spatial object goal 10' for all.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST="${LIBERO_DATA_ROOT:-${DEST:-${1:-}}}"
: "${DEST:?Set LIBERO_DATA_ROOT or DEST to the dataset destination}"
read -r -a suites <<< "${LIBERO_SUITES:-spatial}"
# Validate the complete request before downloading any data.
for suite in "${suites[@]}"; do
  case "${suite}" in
    spatial|object|goal|10) ;;
    *) echo "Unsupported LIBERO suite: ${suite}" >&2; exit 2 ;;
  esac
done
for suite in "${suites[@]}"; do
  dataset="libero_${suite}_no_noops_1.0.0_lerobot"
  hf download "IPEC-COMMUNITY/${dataset}" --repo-type dataset --local-dir "${DEST}/${dataset}"
  mkdir -p "${DEST}/${dataset}/meta"
  cp "${ROOT}/examples/LIBERO/train_files/modality.json" "${DEST}/${dataset}/meta/modality.json"
done
printf 'Dataset root: %s\n' "${DEST}"
