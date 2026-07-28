#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 RUN_DIR" >&2
  exit 2
fi

RUN_DIR="$1"
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/checkpoints"
echo "$BASHPID" > "$RUN_DIR/logs/pipeline.pid"

echo "started_at=$(date --iso-8601=seconds)"
echo "run_dir=$(readlink -f "$RUN_DIR")"
nvidia-smi --query-gpu=name,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits || true

/home/florian/miniforge3/envs/msa_design/bin/python scripts/train_sequence_decoder.py \
  --embeddings-dir outputs/training/okay24_20260713_233827/embeddings \
  --metadata-dir outputs/training/okay24_20260713_233827/metadata \
  --embedding-glob 'ec_*.npz' \
  --max-sequence-length 1024 \
  --d-model 96 \
  --layers 2 \
  --heads 4 \
  --dropout 0.1 \
  --diffusion-timesteps 250 \
  --diffusion-loss-weight 0.5 \
  --token-loss-weight 1.0 \
  --epochs 200 \
  --batch-size 1 \
  --device cuda \
  --numeric-condition-fields kcat_1_per_s,km_mM,kcat_over_km_1_per_mM_s,topt_C,tm_C \
  --categorical-condition-fields ec_numbers,reaction_ids,compound_ids \
  --out-checkpoint "$RUN_DIR/checkpoints/sequence_decoder.pt" \
  --latest-checkpoint "$RUN_DIR/checkpoints/sequence_decoder.latest.pt" \
  --metrics-tsv "$RUN_DIR/logs/metrics.tsv"
status=$?
echo "training_exit=$status"

if [[ -s "$RUN_DIR/logs/metrics.tsv" ]]; then
  /home/florian/miniforge3/envs/msa_design/bin/python scripts/plot_training_curve.py \
    --metrics-tsv "$RUN_DIR/logs/metrics.tsv" \
    --out-svg "$RUN_DIR/logs/training_curve.svg"
  echo "plot_done=$(date --iso-8601=seconds)"
fi

echo "finished_at=$(date --iso-8601=seconds)"
exit "$status"
