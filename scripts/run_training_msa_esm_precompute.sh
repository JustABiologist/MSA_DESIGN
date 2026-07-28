#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/florian/Desktop/MSA_DESIGN"
DATA_ROOT="/mnt/backup4TB/MSA_DESIGN/training_msas_50_identity_core_gaptrim"
OUT_DIR="${DATA_ROOT}/esm_msa_embeddings_col"
LOG="${OUT_DIR}/precompute.log"

if ! findmnt --mountpoint /mnt/backup4TB >/dev/null; then
  echo "ERROR: /mnt/backup4TB is not mounted; refusing to write bulk embeddings to root disk." >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"
cd "${ROOT}"

exec >>"${LOG}" 2>&1

echo "Starting ESM-MSA precompute at $(date -Is)"
exec /home/florian/miniforge3/envs/msa_design/bin/python scripts/precompute_training_msa_embeddings.py \
  --msa-manifest "${DATA_ROOT}/msa_manifest.tsv" \
  --out-dir "${OUT_DIR}" \
  --embedding-manifest "${OUT_DIR}/embedding_manifest.tsv" \
  --weights weights/esm_msa1b_t12_100M_UR50S.pt \
  --device cuda \
  --max-seqs 64 \
  --max-cols 1023 \
  --dtype float16 \
  --progress-every 25 \
  --skip-existing
