#!/usr/bin/env bash
set -Eeuo pipefail
trap 'status=$?; echo "ERROR: auto-start failed at line ${LINENO} status=${status}" >&2' ERR

ROOT="/home/florian/Desktop/MSA_DESIGN"
DATA_ROOT="/mnt/backup4TB/MSA_DESIGN/training_msas_50_identity_core_gaptrim"
TOKEN_DIR="${DATA_ROOT}/esm_msa_token_embeddings_col"
TOKEN_MANIFEST="${TOKEN_DIR}/embedding_manifest.tsv"
SOURCE_MANIFEST="${DATA_ROOT}/msa_manifest.tsv"
WATCH_LOG="${TOKEN_DIR}/auto_start_profile_msa_training.log"

PRECOMPUTE_PID="${PRECOMPUTE_PID:-527671}"
POLL_SECONDS="${POLL_SECONDS:-300}"
EXPECTED_ROWS="${EXPECTED_ROWS:-$(awk -F'\t' 'NR == 1 { for (i = 1; i <= NF; i++) idx[$i] = i; next } { status = (idx["status"] ? $(idx["status"]) : ""); if (status == "ok") n++ } END { print n + 0 }' "${SOURCE_MANIFEST}")}"

manifest_counts() {
  awk -F'\t' '
    NR == 1 {
      for (i = 1; i <= NF; i++) {
        idx[$i] = i
      }
      next
    }
    {
      rows++
      status = (idx["status"] ? $(idx["status"]) : "")
      stores = (idx["stores_token_embeddings"] ? $(idx["stores_token_embeddings"]) : "")
      if (status == "embedded") {
        embedded++
      } else {
        bad++
      }
      if (stores == "True" || stores == "true" || stores == "1") {
        token_true++
      }
    }
    END {
      printf "%d %d %d %d\n", rows + 0, embedded + 0, bad + 0, token_true + 0
    }
  ' "${TOKEN_MANIFEST}"
}

mkdir -p "${TOKEN_DIR}"
cd "${ROOT}"

exec >>"${WATCH_LOG}" 2>&1

echo "auto_start_started_at=$(date -Is)"
echo "root=${ROOT}"
echo "precompute_pid=${PRECOMPUTE_PID}"
echo "token_manifest=${TOKEN_MANIFEST}"
echo "expected_rows=${EXPECTED_ROWS}"
echo "poll_seconds=${POLL_SECONDS}"

while kill -0 "${PRECOMPUTE_PID}" 2>/dev/null; do
  read -r rows embedded bad token_true < <(manifest_counts)
  echo "waiting_at=$(date -Is) rows=${rows}/${EXPECTED_ROWS} embedded=${embedded} bad=${bad} token_true=${token_true}"
  sleep "${POLL_SECONDS}"
done

echo "precompute_pid_exited_at=$(date -Is)"
sleep 15

read -r rows embedded bad token_true < <(manifest_counts)
echo "final_manifest rows=${rows}/${EXPECTED_ROWS} embedded=${embedded} bad=${bad} token_true=${token_true}"

if (( rows != EXPECTED_ROWS || embedded != EXPECTED_ROWS || bad != 0 || token_true != EXPECTED_ROWS )); then
  echo "ERROR: token manifest is not complete and clean; refusing to start training" >&2
  exit 1
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${TRAIN_OUT_DIR:-${DATA_ROOT}/mean_start_ccdd_hardened_grouped_4to5_noaa_esmmsa_tokens_profilemsa_fromscratch_bs1_${STAMP}}"

export EMBED_DIR="${EMBED_DIR:-${TOKEN_DIR}}"
export MEMORY_MODE="${MEMORY_MODE:-profile_msa}"
export PROFILE_FEATURE_MODE="${PROFILE_FEATURE_MODE:-no_aa_frequency}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export MAX_STEPS="${MAX_STEPS:-105178}"
export MASKED_ROWS_PER_MSA_MIN="${MASKED_ROWS_PER_MSA_MIN:-4}"
export MASKED_ROWS_PER_MSA_MAX="${MASKED_ROWS_PER_MSA_MAX:-5}"
export CONSENSUS_LOSS_MODE="${CONSENSUS_LOSS_MODE:-residual}"
export CONSENSUS_MATCH_WEIGHT="${CONSENSUS_MATCH_WEIGHT:-0.35}"
export NONCONSENSUS_WEIGHT="${NONCONSENSUS_WEIGHT:-2.5}"
export UNOBSERVED_NONCONSENSUS_WEIGHT="${UNOBSERVED_NONCONSENSUS_WEIGHT:-1.0}"
export MAX_SEQUENCE_LOSS_WEIGHT="${MAX_SEQUENCE_LOSS_WEIGHT:-3.0}"
export PROFILE_VARIABLE_DROPOUT="${PROFILE_VARIABLE_DROPOUT:-0.0}"
export PROFILE_VARIABLE_BLUR="${PROFILE_VARIABLE_BLUR:-0.0}"

echo "launch_training_at=$(date -Is)"
echo "out_dir=${OUT_DIR}"
echo "embed_dir=${EMBED_DIR}"
echo "memory_mode=${MEMORY_MODE}"
echo "profile_feature_mode=${PROFILE_FEATURE_MODE}"
echo "batch_size=${BATCH_SIZE}"
echo "max_steps=${MAX_STEPS}"
echo "masked_rows_per_msa=${MASKED_ROWS_PER_MSA_MIN}:${MASKED_ROWS_PER_MSA_MAX}"
echo "consensus_loss_mode=${CONSENSUS_LOSS_MODE}"

exec "${ROOT}/scripts/run_mean_start_ccdd_full_profile_row.sh" "${OUT_DIR}"
