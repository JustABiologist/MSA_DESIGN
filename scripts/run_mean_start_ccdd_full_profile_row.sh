#!/usr/bin/env bash
set -Eeuo pipefail
trap 'status=$?; echo "ERROR: launcher failed at line ${LINENO} status=${status}" >&2' ERR

ROOT="/home/florian/Desktop/MSA_DESIGN"
DEFAULT_DATA_ROOT="/mnt/backup4TB/MSA_DESIGN/training_msas_50_identity_core_gaptrim"
DATA_ROOT="${DATA_ROOT:-${DEFAULT_DATA_ROOT}}"
EMBED_DIR="${EMBED_DIR:-${DATA_ROOT}/esm_msa_embeddings_col}"
LABEL_SUMMARY="${DATA_ROOT}/sequence_label_summary.tsv.gz"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${1:-${DATA_ROOT}/mean_start_ccdd_full_profile_row_${STAMP}}"
TRAIN_LOG="${OUT_DIR}/train.log"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_STEPS="${MAX_STEPS:-50000}"
MEMORY_MODE="${MEMORY_MODE:-profile_row}"
D_MODEL="${D_MODEL:-192}"
CONTINUOUS_TARGET_MODE="${CONTINUOUS_TARGET_MODE:-target_row_embedding}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
RESET_OPTIMIZER="${RESET_OPTIMIZER:-0}"
ALLOW_PARTIAL_RESUME="${ALLOW_PARTIAL_RESUME:-0}"
MASKED_ROWS_PER_MSA_MIN="${MASKED_ROWS_PER_MSA_MIN:-1}"
MASKED_ROWS_PER_MSA_MAX="${MASKED_ROWS_PER_MSA_MAX:-1}"
PROFILE_FEATURE_MODE="${PROFILE_FEATURE_MODE:-full}"
CONDITION_MASK_PROB="${CONDITION_MASK_PROB:-0.25}"
NUMERIC_CONDITION_LOSS_WEIGHT="${NUMERIC_CONDITION_LOSS_WEIGHT:-0.2}"
CATEGORY_CONDITION_LOSS_WEIGHT="${CATEGORY_CONDITION_LOSS_WEIGHT:-0.02}"
CONDITION_PRESENCE_LOSS_WEIGHT="${CONDITION_PRESENCE_LOSS_WEIGHT:-0.05}"
CONSENSUS_LOSS_MODE="${CONSENSUS_LOSS_MODE:-none}"
CONSENSUS_MATCH_WEIGHT="${CONSENSUS_MATCH_WEIGHT:-0.35}"
NONCONSENSUS_WEIGHT="${NONCONSENSUS_WEIGHT:-2.5}"
UNOBSERVED_NONCONSENSUS_WEIGHT="${UNOBSERVED_NONCONSENSUS_WEIGHT:-1.0}"
MAX_SEQUENCE_LOSS_WEIGHT="${MAX_SEQUENCE_LOSS_WEIGHT:-3.0}"
VARIABLE_COLUMN_MIN_ENTROPY="${VARIABLE_COLUMN_MIN_ENTROPY:-0.05}"
VARIABLE_COLUMN_MAX_CONSENSUS="${VARIABLE_COLUMN_MAX_CONSENSUS:-0.92}"
PROFILE_VARIABLE_DROPOUT="${PROFILE_VARIABLE_DROPOUT:-0.0}"
PROFILE_VARIABLE_BLUR="${PROFILE_VARIABLE_BLUR:-0.0}"
PROFILE_BLUR_ALPHA="${PROFILE_BLUR_ALPHA:-0.5}"
MSA_EMBEDDING_DTYPE="${MSA_EMBEDDING_DTYPE:-float32}"
AMP="${AMP:-off}"
MSA_AXIAL_LAYERS="${MSA_AXIAL_LAYERS:-1}"
MAX_MSA_CONTEXT_ROWS="${MAX_MSA_CONTEXT_ROWS:-}"
MIXED_MSA_CONTEXT_ROWS="${MIXED_MSA_CONTEXT_ROWS:-0}"
CACHE_SIZE="${CACHE_SIZE:-128}"
VAL_BATCHES="${VAL_BATCHES:-64}"
CHECKPOINT_EVERY_STEPS="${CHECKPOINT_EVERY_STEPS:-1000}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF

PATH_REWRITE="${PATH_REWRITE:-}"
if [[ -z "${PATH_REWRITE}" && "${DATA_ROOT}" != "${DEFAULT_DATA_ROOT}" ]]; then
  DATA_PREFIX="${DATA_ROOT%/MSA_DESIGN/training_msas_50_identity_core_gaptrim}"
  if [[ "${DATA_PREFIX}" != "${DATA_ROOT}" ]]; then
    PATH_REWRITE="/mnt/backup4TB=${DATA_PREFIX}"
  fi
fi

if [[ "${DATA_ROOT}" == /mnt/backup4TB || "${DATA_ROOT}" == /mnt/backup4TB/* ]] && ! findmnt --mountpoint /mnt/backup4TB >/dev/null; then
  echo "ERROR: /mnt/backup4TB is not mounted; refusing to train from missing 4TB data." >&2
  exit 1
fi
if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "ERROR: DATA_ROOT does not exist: ${DATA_ROOT}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"
cd "${ROOT}"

exec >>"${TRAIN_LOG}" 2>&1

echo "Mean-start CCDD profile/MSA/row training launcher"
echo "started_at=$(date -Is)"
echo "root=${ROOT}"
echo "data_root=${DATA_ROOT}"
echo "out_dir=${OUT_DIR}"
echo "embedding_manifest=${EMBED_DIR}/embedding_manifest.tsv"
echo "label_summary=${LABEL_SUMMARY}"
echo "runner_pid=$$"
echo "batch_size=${BATCH_SIZE}"
echo "max_steps=${MAX_STEPS}"
echo "memory_mode=${MEMORY_MODE}"
echo "d_model=${D_MODEL}"
echo "continuous_target_mode=${CONTINUOUS_TARGET_MODE}"
echo "resume_checkpoint=${RESUME_CHECKPOINT}"
echo "reset_optimizer=${RESET_OPTIMIZER}"
echo "allow_partial_resume=${ALLOW_PARTIAL_RESUME}"
echo "masked_rows_per_msa_min=${MASKED_ROWS_PER_MSA_MIN}"
echo "masked_rows_per_msa_max=${MASKED_ROWS_PER_MSA_MAX}"
echo "profile_feature_mode=${PROFILE_FEATURE_MODE}"
echo "condition_mask_prob=${CONDITION_MASK_PROB}"
echo "numeric_condition_loss_weight=${NUMERIC_CONDITION_LOSS_WEIGHT}"
echo "category_condition_loss_weight=${CATEGORY_CONDITION_LOSS_WEIGHT}"
echo "condition_presence_loss_weight=${CONDITION_PRESENCE_LOSS_WEIGHT}"
echo "consensus_loss_mode=${CONSENSUS_LOSS_MODE}"
echo "consensus_match_weight=${CONSENSUS_MATCH_WEIGHT}"
echo "nonconsensus_weight=${NONCONSENSUS_WEIGHT}"
echo "unobserved_nonconsensus_weight=${UNOBSERVED_NONCONSENSUS_WEIGHT}"
echo "max_sequence_loss_weight=${MAX_SEQUENCE_LOSS_WEIGHT}"
echo "variable_column_min_entropy=${VARIABLE_COLUMN_MIN_ENTROPY}"
echo "variable_column_max_consensus=${VARIABLE_COLUMN_MAX_CONSENSUS}"
echo "profile_variable_dropout=${PROFILE_VARIABLE_DROPOUT}"
echo "profile_variable_blur=${PROFILE_VARIABLE_BLUR}"
echo "profile_blur_alpha=${PROFILE_BLUR_ALPHA}"
echo "msa_embedding_dtype=${MSA_EMBEDDING_DTYPE}"
echo "amp=${AMP}"
echo "msa_axial_layers=${MSA_AXIAL_LAYERS}"
echo "max_msa_context_rows=${MAX_MSA_CONTEXT_ROWS}"
echo "mixed_msa_context_rows=${MIXED_MSA_CONTEXT_ROWS}"
echo "cache_size=${CACHE_SIZE}"
echo "val_batches=${VAL_BATCHES}"
echo "checkpoint_every_steps=${CHECKPOINT_EVERY_STEPS}"
echo "path_rewrite=${PATH_REWRITE}"
echo "pytorch_cuda_alloc_conf=${PYTORCH_CUDA_ALLOC_CONF}"
echo "$$" > "${OUT_DIR}/runner.pid"

gzip -t "${LABEL_SUMMARY}"
echo "label_summary_gzip_ok=1"
test -s "${EMBED_DIR}/embedding_manifest.tsv"
echo "embedding_manifest_ok=1"
CMD=(
  /home/florian/miniforge3/envs/msa_design/bin/python scripts/train_mean_start_ccdd_from_cached_msas.py
  --embedding-manifest "${EMBED_DIR}/embedding_manifest.tsv" \
  --label-summary "${LABEL_SUMMARY}" \
  --out-dir "${OUT_DIR}" \
  --memory-mode "${MEMORY_MODE}" \
  --profile-feature-mode "${PROFILE_FEATURE_MODE}" \
  --batch-size "${BATCH_SIZE}" \
  --d-model "${D_MODEL}" \
  --layers 4 \
  --heads 6 \
  --msa-axial-layers "${MSA_AXIAL_LAYERS}" \
  --diffusion-timesteps 250 \
  --max-diffusion-timestep 249 \
  --decoder-start-mode noisy_mean \
  --continuous-target-mode "${CONTINUOUS_TARGET_MODE}" \
  --continuous-loss-weight 0.5 \
  --token-loss-weight 1.0 \
  --consensus-loss-mode "${CONSENSUS_LOSS_MODE}" \
  --consensus-match-weight "${CONSENSUS_MATCH_WEIGHT}" \
  --nonconsensus-weight "${NONCONSENSUS_WEIGHT}" \
  --unobserved-nonconsensus-weight "${UNOBSERVED_NONCONSENSUS_WEIGHT}" \
  --max-sequence-loss-weight "${MAX_SEQUENCE_LOSS_WEIGHT}" \
  --variable-column-min-entropy "${VARIABLE_COLUMN_MIN_ENTROPY}" \
  --variable-column-max-consensus "${VARIABLE_COLUMN_MAX_CONSENSUS}" \
  --profile-variable-dropout "${PROFILE_VARIABLE_DROPOUT}" \
  --profile-variable-blur "${PROFILE_VARIABLE_BLUR}" \
  --profile-blur-alpha "${PROFILE_BLUR_ALPHA}" \
  --condition-mask-prob "${CONDITION_MASK_PROB}" \
  --numeric-condition-loss-weight "${NUMERIC_CONDITION_LOSS_WEIGHT}" \
  --category-condition-loss-weight "${CATEGORY_CONDITION_LOSS_WEIGHT}" \
  --condition-presence-loss-weight "${CONDITION_PRESENCE_LOSS_WEIGHT}" \
  --max-steps "${MAX_STEPS}" \
  --msa-embedding-dtype "${MSA_EMBEDDING_DTYPE}" \
  --amp "${AMP}" \
  --masked-rows-per-msa-min "${MASKED_ROWS_PER_MSA_MIN}" \
  --masked-rows-per-msa-max "${MASKED_ROWS_PER_MSA_MAX}" \
  --mixed-msa-context-rows "${MIXED_MSA_CONTEXT_ROWS}" \
  --log-every-steps 25 \
  --eval-every-steps 500 \
  --val-batches "${VAL_BATCHES}" \
  --decode-every-steps 500 \
  --decode-examples 12 \
  --checkpoint-every-steps "${CHECKPOINT_EVERY_STEPS}" \
  --cache-size "${CACHE_SIZE}" \
  --device cuda
)

if [[ -n "${PATH_REWRITE}" ]]; then
  CMD+=(--path-rewrite "${PATH_REWRITE}")
fi
if [[ -n "${MAX_MSA_CONTEXT_ROWS}" ]]; then
  CMD+=(--max-msa-context-rows "${MAX_MSA_CONTEXT_ROWS}")
fi

if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  CMD+=(--resume-checkpoint "${RESUME_CHECKPOINT}")
fi
if [[ "${RESET_OPTIMIZER}" == "1" ]]; then
  CMD+=(--reset-optimizer)
fi
if [[ "${ALLOW_PARTIAL_RESUME}" == "1" ]]; then
  CMD+=(--allow-partial-resume)
fi

nvidia-smi || true
printf 'command:'
printf ' %q' "${CMD[@]}"
printf '\n'

exec "${CMD[@]}"
