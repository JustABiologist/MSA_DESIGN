#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/florian/Desktop/MSA_DESIGN"
DATA_ROOT="/mnt/backup4TB/MSA_DESIGN/training_msas_50_identity_core_gaptrim"
EMBED_DIR="${DATA_ROOT}/esm_msa_embeddings_col"
EMBED_LOG="${EMBED_DIR}/precompute.log"
LABEL_SUMMARY="${DATA_ROOT}/sequence_label_summary.tsv.gz"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${DATA_ROOT}/aligned_column_training_full_${STAMP}"
TRAIN_LOG="${OUT_DIR}/train.log"

if ! findmnt --mountpoint /mnt/backup4TB >/dev/null; then
  echo "ERROR: /mnt/backup4TB is not mounted; refusing to train from missing 4TB data." >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"
cd "${ROOT}"

exec >>"${TRAIN_LOG}" 2>&1

echo "Watcher started at $(date -Is)"
echo "Waiting for ESM-MSA precompute to finish before starting GPU training."
while pgrep -f "precompute_training_msa_embeddings.py .*${EMBED_DIR}" >/dev/null; do
  tail -n 3 "${EMBED_LOG}" || true
  sleep 300
done

if ! grep -q "Done embedding .*failed=0" "${EMBED_LOG}"; then
  echo "ERROR: ESM-MSA precompute did not finish cleanly with failed=0. Not starting training."
  tail -n 40 "${EMBED_LOG}" || true
  exit 1
fi

gzip -t "${LABEL_SUMMARY}"
echo "Starting aligned-column training at $(date -Is)"

exec /home/florian/miniforge3/envs/msa_design/bin/python scripts/train_aligned_column_decoder.py \
  --embedding-manifest "${EMBED_DIR}/embedding_manifest.tsv" \
  --label-summary "${LABEL_SUMMARY}" \
  --out-dir "${OUT_DIR}" \
  --batch-size 8 \
  --d-model 192 \
  --layers 4 \
  --heads 6 \
  --max-steps 20000 \
  --log-every-steps 25 \
  --eval-every-steps 500 \
  --val-batches 64 \
  --checkpoint-every-steps 250 \
  --gap-loss-weight 0.5 \
  --cache-size 128 \
  --device cuda
