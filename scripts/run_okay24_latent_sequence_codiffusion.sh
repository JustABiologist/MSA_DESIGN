#!/usr/bin/env bash
set -Eeuo pipefail

RUN_DIR="${1:-outputs/training/okay24_latent_sequence_codiffusion_$(date +%Y%m%d_%H%M%S)}"
if [[ $# -gt 0 ]]; then
  shift
fi
INIT_CHECKPOINT="${INIT_CHECKPOINT:-outputs/training/okay24_noisy_mean_curriculum_20260715_073407/checkpoints/sequence_decoder.latest.pt}"

exec bash scripts/run_okay24_hard_regime_single.sh "$RUN_DIR" \
  --epochs "${EPOCHS:-300}" \
  --init-checkpoint "$INIT_CHECKPOINT" \
  --min-diffusion-timestep "${MIN_DIFFUSION_TIMESTEP:-200}" \
  --max-diffusion-timestep "${MAX_DIFFUSION_TIMESTEP:-249}" \
  --timestep-curriculum-epochs "${TIMESTEP_CURRICULUM_EPOCHS:-150}" \
  --curriculum-start-min-diffusion-timestep "${CURRICULUM_START_MIN_DIFFUSION_TIMESTEP:-0}" \
  --curriculum-start-max-diffusion-timestep "${CURRICULUM_START_MAX_DIFFUSION_TIMESTEP:-49}" \
  --decoder-start-mode noisy_mean \
  --latent-codiffusion-tokens "${LATENT_CODIFFUSION_TOKENS:-32}" \
  --diffusion-loss-weight "${DIFFUSION_LOSS_WEIGHT:-0.5}" \
  --latent-loss-weight "${LATENT_LOSS_WEIGHT:-0.5}" \
  --token-loss-weight "${TOKEN_LOSS_WEIGHT:-1.0}" \
  --condition-dropout "${CONDITION_DROPOUT:-0.10}" \
  "$@"
