#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MMSEQS="${MMSEQS:-$PROJECT_ROOT/tools/mmseqs/bin/mmseqs}"
INPUT_FASTA="${INPUT_FASTA:-$PROJECT_ROOT/outputs/kegg_uniprot/all_sequences.fasta.gz}"
MOUNT_POINT="${MOUNT_POINT:-/mnt/backup4TB}"
WORK_ROOT="${WORK_ROOT:-$MOUNT_POINT/MSA_DESIGN/kegg_uniprot_global_identity}"
THREADS="${THREADS:-24}"
THRESHOLDS="${THRESHOLDS:-0.90 0.70 0.50 0.30}"
COVERAGE="${COVERAGE:-0.80}"
MIN_GOOD_SIZE="${MIN_GOOD_SIZE:-16}"
MAX_GOOD_SIZE="${MAX_GOOD_SIZE:-4096}"
SPLIT_MEMORY_LIMIT="${SPLIT_MEMORY_LIMIT:-48G}"

if ! command -v findmnt >/dev/null 2>&1; then
  echo "findmnt is required to verify the 4TB mount." >&2
  exit 2
fi

if ! findmnt --mountpoint "$MOUNT_POINT" >/dev/null 2>&1; then
  echo "$MOUNT_POINT is not mounted. Mount the 4TB disk before running this." >&2
  exit 2
fi

if [[ ! -x "$MMSEQS" ]]; then
  echo "MMseqs binary not found or not executable: $MMSEQS" >&2
  exit 2
fi

if [[ ! -s "$INPUT_FASTA" ]]; then
  echo "Input FASTA not found: $INPUT_FASTA" >&2
  exit 2
fi

mkdir -p "$WORK_ROOT"/{db,tmp,logs,clusters}

DB="$WORK_ROOT/db/all_sequences"
if [[ ! -s "$DB.dbtype" ]]; then
  "$MMSEQS" createdb "$INPUT_FASTA" "$DB" \
    |& tee "$WORK_ROOT/logs/createdb.log"
fi

printf "threshold\tcluster_tsv\tcluster_stats\tsummary\tgood_clusters\n" > "$WORK_ROOT/clustering_outputs.tsv"

for threshold in $THRESHOLDS; do
  tag="${threshold/./}"
  OUT_DIR="$WORK_ROOT/clusters/id_${tag}_cov_${COVERAGE/./}"
  TMP_DIR="$WORK_ROOT/tmp/id_${tag}_cov_${COVERAGE/./}"
  CLU="$OUT_DIR/clu"
  CLUSTER_TSV="$OUT_DIR/clusters.tsv"
  CLUSTER_STATS="$OUT_DIR/cluster_stats.tsv"
  SUMMARY="$OUT_DIR/cluster_summary.tsv"
  GOOD_CLUSTERS="$OUT_DIR/good_msa_clusters.tsv"
  GOOD_MEMBERS="$OUT_DIR/good_msa_members.tsv"
  LOG="$WORK_ROOT/logs/linclust_id_${tag}_cov_${COVERAGE/./}.log"

  mkdir -p "$OUT_DIR" "$TMP_DIR"
  if [[ ! -s "$CLU.dbtype" ]]; then
    "$MMSEQS" linclust "$DB" "$CLU" "$TMP_DIR" \
      --min-seq-id "$threshold" \
      -c "$COVERAGE" \
      --cov-mode 0 \
      --seq-id-mode 2 \
      --alignment-mode 3 \
      --threads "$THREADS" \
      --split-memory-limit "$SPLIT_MEMORY_LIMIT" \
      --remove-tmp-files 1 \
      |& tee "$LOG"
  fi

  if [[ ! -s "$CLUSTER_TSV" ]]; then
    "$MMSEQS" createtsv "$DB" "$DB" "$CLU" "$CLUSTER_TSV" \
      |& tee "$WORK_ROOT/logs/createtsv_id_${tag}_cov_${COVERAGE/./}.log"
  fi

  python3 "$PROJECT_ROOT/scripts/summarize_mmseqs_clusters.py" \
    --clusters "$CLUSTER_TSV" \
    --cluster-stats "$CLUSTER_STATS" \
    --summary "$SUMMARY" \
    --good-clusters "$GOOD_CLUSTERS" \
    --good-members "$GOOD_MEMBERS" \
    --min-size "$MIN_GOOD_SIZE" \
    --max-size "$MAX_GOOD_SIZE"

  printf "%s\t%s\t%s\t%s\t%s\n" \
    "$threshold" "$CLUSTER_TSV" "$CLUSTER_STATS" "$SUMMARY" "$GOOD_CLUSTERS" \
    >> "$WORK_ROOT/clustering_outputs.tsv"
done

echo "Global identity clustering complete: $WORK_ROOT"
