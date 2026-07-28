#!/usr/bin/env bash
set -Eeuo pipefail

RUN_DIR="${1:-outputs/training/okay24_discrete_sequence_diffusion_$(date +%Y%m%d_%H%M%S)}"
if [[ $# -gt 0 ]]; then
  shift
fi
INIT_CHECKPOINT="${INIT_CHECKPOINT:-outputs/training/okay24_hard_regime_sweep_20260714_174848/02_mean_start_t249/checkpoints/sequence_decoder.latest.pt}"

exec bash scripts/run_okay24_hard_regime_single.sh "$RUN_DIR" \
  --epochs "${EPOCHS:-300}" \
  --init-checkpoint "$INIT_CHECKPOINT" \
  --min-diffusion-timestep 0 \
  --max-diffusion-timestep 249 \
  --decoder-start-mode discrete_mask \
  --diffusion-loss-weight 0.0 \
  --token-loss-weight 1.0 \
  --condition-dropout "${CONDITION_DROPOUT:-0.10}" \
  --discrete-loss-corrupted-only \
  "$@"
