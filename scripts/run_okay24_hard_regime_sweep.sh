#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 RUN_ROOT" >&2
  exit 2
fi

RUN_ROOT="$1"
mkdir -p "$RUN_ROOT/logs"
echo "$BASHPID" > "$RUN_ROOT/logs/sweep.pid"

COMMON_ARGS=(
  --embeddings-dir outputs/training/okay24_20260713_233827/embeddings
  --metadata-dir outputs/training/okay24_20260713_233827/metadata
  --embedding-glob 'ec_*.npz'
  --max-sequence-length 1024
  --d-model 96
  --layers 2
  --heads 4
  --dropout 0.1
  --diffusion-timesteps 250
  --diffusion-loss-weight 0.5
  --token-loss-weight 1.0
  --epochs 50
  --batch-size 1
  --lr 5e-5
  --device cuda
  --mask-target-row-in-msa
  --init-checkpoint outputs/training/okay24_numeric_leaveoneout_20260714_144815/checkpoints/sequence_decoder.latest.pt
  --numeric-condition-fields kcat_1_per_s,km_mM,kcat_over_km_1_per_mM_s,topt_C,tm_C
  --categorical-condition-fields ec_numbers,reaction_ids,compound_ids
)

run_regime() {
  local name="$1"
  shift
  local run_dir="$RUN_ROOT/$name"
  mkdir -p "$run_dir/logs" "$run_dir/checkpoints"
  echo "$BASHPID" > "$run_dir/logs/pipeline.pid"
  {
    echo "started_at=$(date --iso-8601=seconds)"
    echo "run_dir=$(readlink -f "$run_dir")"
    echo "regime=$name"
    nvidia-smi --query-gpu=name,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits || true
    /home/florian/miniforge3/envs/msa_design/bin/python scripts/train_sequence_decoder.py \
      "${COMMON_ARGS[@]}" \
      "$@" \
      --out-checkpoint "$run_dir/checkpoints/sequence_decoder.pt" \
      --latest-checkpoint "$run_dir/checkpoints/sequence_decoder.latest.pt" \
      --metrics-tsv "$run_dir/logs/metrics.tsv"
    local status=$?
    echo "training_exit=$status"
    if [[ -s "$run_dir/logs/metrics.tsv" ]]; then
      /home/florian/miniforge3/envs/msa_design/bin/python scripts/plot_training_curve.py \
        --metrics-tsv "$run_dir/logs/metrics.tsv" \
        --out-svg "$run_dir/logs/training_curve.svg"
      echo "plot_done=$(date --iso-8601=seconds)"
    fi
    echo "finished_at=$(date --iso-8601=seconds)"
    exit "$status"
  } > "$run_dir/logs/pipeline.log" 2>&1
}

{
  echo "sweep_started_at=$(date --iso-8601=seconds)"
  echo "run_root=$(readlink -f "$RUN_ROOT")"

  run_regime 01_high_t200_249 \
    --min-diffusion-timestep 200 \
    --max-diffusion-timestep 249 \
    --decoder-start-mode q_sample

  run_regime 02_mean_start_t249 \
    --min-diffusion-timestep 249 \
    --max-diffusion-timestep 249 \
    --decoder-start-mode mean

  run_regime 03_span_dropout_qsample \
    --decoder-start-mode q_sample \
    --decoder-token-dropout 0.25 \
    --decoder-span-mask-fraction 0.40 \
    --decoder-span-mask-length 32

  echo "sweep_finished_at=$(date --iso-8601=seconds)"
} > "$RUN_ROOT/logs/sweep.log" 2>&1
