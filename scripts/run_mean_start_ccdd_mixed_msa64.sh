#!/usr/bin/env bash
set -Eeuo pipefail
trap 'status=$?; echo "ERROR: mixed-MSA launcher failed at line ${LINENO} status=${status}" >&2' ERR

ROOT="/home/florian/Desktop/MSA_DESIGN"
DEFAULT_DATA_ROOT="/media/florian/dc434ad3-38cd-442f-b7af-41802cfa5baa/MSA_DESIGN/training_msas_50_identity_core_gaptrim"
DATA_ROOT="${DATA_ROOT:-${DEFAULT_DATA_ROOT}}"
STAMP="$(date +%Y%m%d_%H%M%S)"
ORGANISM_RUN_DIR="${ORGANISM_RUN_DIR:-${DATA_ROOT}/mean_start_ccdd_organismcode_condtokens_targetrow_latents_direct_axial_reads_sharedgrid_grouped_4to5_noaa_esmmsa_tokens_fullgrid_fp16amp_residual_condtokens_unmasked_partialresume_frombest173500_20260805_211457}"
OUT_DIR="${1:-${DATA_ROOT}/mean_start_ccdd_mixed64msas_targetrow_latents_direct_axial_reads_noaa_esmmsa_tokens_fullgrid_fp16amp_residual_resume_from_organismbest_${STAMP}}"

if [[ "${ALLOW_CONCURRENT_CUDA_RUN:-0}" != "1" ]]; then
  RUNNING_TRAIN_PIDS="$(pgrep -f 'scripts/train_mean_start_ccdd_from_cached_msas.py' || true)"
  if [[ -n "${RUNNING_TRAIN_PIDS}" ]]; then
    echo "ERROR: another mean-start CCDD trainer is already running: ${RUNNING_TRAIN_PIDS}" >&2
    echo "Set ALLOW_CONCURRENT_CUDA_RUN=1 only if you really want to compete for CUDA memory." >&2
    exit 1
  fi
fi

export DATA_ROOT
export EMBED_DIR="${EMBED_DIR:-${DATA_ROOT}/esm_msa_token_embeddings_col}"
export MEMORY_MODE="${MEMORY_MODE:-profile_msa_axial}"
export PROFILE_FEATURE_MODE="${PROFILE_FEATURE_MODE:-no_aa_frequency}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export D_MODEL="${D_MODEL:-192}"
export CONTINUOUS_TARGET_MODE="${CONTINUOUS_TARGET_MODE:-target_row_embedding}"
export MAX_STEPS="${MAX_STEPS:-544000}"
export MASKED_ROWS_PER_MSA_MIN="${MASKED_ROWS_PER_MSA_MIN:-1}"
export MASKED_ROWS_PER_MSA_MAX="${MASKED_ROWS_PER_MSA_MAX:-1}"
export MIXED_MSA_CONTEXT_ROWS="${MIXED_MSA_CONTEXT_ROWS:-64}"
export CONSENSUS_LOSS_MODE="${CONSENSUS_LOSS_MODE:-residual}"
export CONSENSUS_MATCH_WEIGHT="${CONSENSUS_MATCH_WEIGHT:-0.35}"
export NONCONSENSUS_WEIGHT="${NONCONSENSUS_WEIGHT:-2.5}"
export UNOBSERVED_NONCONSENSUS_WEIGHT="${UNOBSERVED_NONCONSENSUS_WEIGHT:-1.0}"
export MAX_SEQUENCE_LOSS_WEIGHT="${MAX_SEQUENCE_LOSS_WEIGHT:-3.0}"
export CONDITION_MASK_PROB="${CONDITION_MASK_PROB:-0.0}"
export NUMERIC_CONDITION_LOSS_WEIGHT="${NUMERIC_CONDITION_LOSS_WEIGHT:-0.2}"
export CATEGORY_CONDITION_LOSS_WEIGHT="${CATEGORY_CONDITION_LOSS_WEIGHT:-0.02}"
export CONDITION_PRESENCE_LOSS_WEIGHT="${CONDITION_PRESENCE_LOSS_WEIGHT:-0.05}"
export MSA_EMBEDDING_DTYPE="${MSA_EMBEDDING_DTYPE:-float16}"
export AMP="${AMP:-fp16}"
export MSA_AXIAL_LAYERS="${MSA_AXIAL_LAYERS:-1}"
export CACHE_SIZE="${CACHE_SIZE:-96}"
export VAL_BATCHES="${VAL_BATCHES:-32}"
export CHECKPOINT_EVERY_STEPS="${CHECKPOINT_EVERY_STEPS:-500}"
export RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-${ORGANISM_RUN_DIR}/mean_start_ccdd.best.pt}"
export RESET_OPTIMIZER="${RESET_OPTIMIZER:-1}"
export ALLOW_PARTIAL_RESUME="${ALLOW_PARTIAL_RESUME:-1}"

if [[ ! -s "${RESUME_CHECKPOINT}" ]]; then
  echo "ERROR: resume checkpoint missing or empty: ${RESUME_CHECKPOINT}" >&2
  exit 1
fi

exec "${ROOT}/scripts/run_mean_start_ccdd_full_profile_row.sh" "${OUT_DIR}"
