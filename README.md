# MSA_DESIGN

MSA_DESIGN is the local research stack for enzyme sequence design from frozen
ESM-MSA embeddings and GotEnzymes reaction metadata. The current workhorse is a
mean-start CCDD-style target-row reconstruction model that keeps a
target-masked MSA token grid static and lets only the sequence decoder latents
read from it.

The older pilot scripts are still useful for data checks and baselines, but the
active architecture is `profile_msa_axial` in
`scripts/train_mean_start_ccdd_from_cached_msas.py`.

## Current Architecture

The active model reconstructs one or more hidden target rows from an enzyme MSA
under numeric and categorical enzyme conditions.

Inputs per training example:

- A target protein sequence encoded as residues, one `*` STOP token, then low
  weight trailing `*` padding.
- A leave-one-target-row-out profile built from the remaining aligned rows.
  Current high-signal runs usually use `--profile-feature-mode no_aa_frequency`,
  which removes amino-acid frequencies and keeps only gap/coverage channels.
- Numeric condition fields: `kcat_1_per_s`, `km_mM`,
  `kcat_over_km_1_per_mM_s`, `topt_C`, and `tm_C`.
- Categorical condition fields used by the current mean-start trainer:
  `domain`, `reaction_id`, `ec_numbers`, `compound_id`, and `organism_code`.
- Cached ESM-MSA token embeddings for the retained MSA rows, shaped
  `rows x aligned_columns x 768`. The active token cache is
  `esm_msa_token_embeddings_col/embedding_manifest.tsv`.
- Optional target-row ESM-MSA residue embeddings, extracted before masking and
  used only as continuous supervision.

CCDD wording in this repo is precise:

- The active mean-start trainer is CCDD-inspired dual supervision, not a full
  paper-faithful latent+token co-diffusion loop. The sequence stream starts from
  mean or noisy-mean amino-acid embeddings and predicts residue tokens with CE.
- In `--continuous-target-mode target_row_embedding`, the same decoder states
  also predict the hidden target row's frozen ESM-MSA per-residue embeddings
  with an MSE-style continuous loss. This is the "latent" side of the current
  run.
- Those target-row continuous embeddings are not provided as conditioning
  memory and are not allowed to leak through the MSA grid. They are extracted
  before target-row masking only so they can serve as the supervised target.
- The older `scripts/train_sequence_decoder.py` path contains the more explicit
  CCDD-lite experiment: discrete absorbing-mask corruption plus a noised
  continuous target stream, with masked discrete positions erased in the
  continuous stream before Gaussian noising. That path is kept as a legacy
  baseline, while current large runs use `train_mean_start_ccdd_from_cached_msas.py`.

Leakage controls:

- Target row(s) are removed from the profile before profile construction.
- Target row(s) are removed from cached ESM-MSA token-grid memory.
- Target row(s) are removed from row-memory paths in legacy modes.
- Cached `col_embeddings` are not used in the active `profile_msa_axial` path.
- Target-row residue embeddings are allowed only as the continuous target when
  `--continuous-target-mode target_row_embedding` is selected.

Decoder flow per `SequenceMSAAxialDecoderLayer`:

1. Self-attention over the sequence decoder latents.
2. Cross-attention from sequence latents into static condition/profile memory.
3. Direct same-column MSA read: decoder position `i` attends to all retained
   MSA row cells at aligned column `i`.
4. Direct whole-row MSA read: decoder latents attend over every retained MSA row
   sequence, then fuse row-specific updates with attention.
5. Feed-forward update on the sequence latents.

The MSA grid is read-only in this path. The layer returns `sequence, msa_grid`,
with `msa_grid` unchanged. Do not replace these direct column and row reads with
pooled row/column summaries unless that is the experiment being tested.

Grouped-target mode reconstructs 4 to 5 rows from one MSA while sharing the
heavy MSA grid. The collator emits one shared grid per MSA group plus
`target_msa_group_indices`; each target keeps its own condition/profile tensors
while `_column_read` and `_row_read` index into the shared static grid.

Mixed-MSA context mode is a harder negative-control style experiment. When
`--mixed-msa-context-rows N` is set, each target uses a static axial grid
assembled from one row sampled from each of `N` different MSA groups. The target
MSA is excluded from that grid, and this mode currently requires
`--masked-rows-per-msa-min 1 --masked-rows-per-msa-max 1`.

## Data And Artifacts

Large data, model weights, checkpoints, and generated outputs are intentionally
ignored by Git. Important local artifacts:

- `data/input_data.zip`, Zenodo GotEnzymes archive.
- `weights/esm_msa1b_t12_100M_UR50S.pt`, local ESM-MSA-1b weights.
- `weights/esm_msa1b_t12_100M_UR50S-contact-regression.pt`, the small
  `fair-esm` sidecar expected by local loading.
- `/mnt/backup4TB/MSA_DESIGN/training_msas_50_identity_core_gaptrim/`, the
  canonical training root encoded in manifests.
- On this workstation the same root is often mounted at
  `/media/florian/dc434ad3-38cd-442f-b7af-41802cfa5baa/MSA_DESIGN/training_msas_50_identity_core_gaptrim/`.
  Launchers set a `/mnt/backup4TB=...` path rewrite automatically when
  `DATA_ROOT` points at the `/media/...` mount.

The current low-gap UniProt-backed training MSA set contains:

- 31,727 successful heavily gap-trimmed MSAs from 33,194 input clusters.
- 1,915,502 kept sequences.
- 13,741,668 retained reaction-parameter rows.
- Mean raw gap fraction 0.121171 and mean trimmed gap fraction 0.004481.

Key files in the training root:

- `msa_manifest.tsv`, one row per training MSA build attempt.
- `sequence_manifest.tsv.gz`, retained sequence-to-cluster membership.
- `kept_sequence_index.tsv.gz`, sequence index for the kept subset.
- `kept_reaction_parameters.tsv.gz`, GotEnzymes reaction rows for kept entries.
- `sequence_label_summary.tsv.gz`, per-sequence numeric and categorical labels.
- `trimmed_alignments/`, final aligned FASTA files.
- `esm_msa_embeddings_col/`, pooled ESM-MSA embeddings for older baselines.
- `esm_msa_token_embeddings_col/`, active token-grid cache with
  `token_embeddings` stored.

ESM-MSA accepts at most 1024 tokens including its special token, so training
embedding precompute uses `--max-cols 1023`.

## Environment

Use the dedicated CUDA conda environment:

```bash
/home/florian/miniforge3/bin/mamba env create -f environment.yml
conda activate msa_design
```

Without shell activation, run commands through the environment directly:

```bash
/home/florian/miniforge3/envs/msa_design/bin/python scripts/train_mean_start_ccdd_from_cached_msas.py --help
```

The environment includes Python 3.10, CUDA PyTorch, Kalign, NumPy, and
`fair-esm`. Local CUDA tests have used the RTX 3060.

## GotEnzymes Framing

`input_data/enzymes/*.txt` rows have 11 tab-separated columns:

1. `gene_id`
2. `organism_code`
3. `domain`
4. `reaction_id`
5. `ec_numbers`
6. `compound_id`
7. `kcat_1_per_s`
8. `km_mM`
9. `kcat_over_km_1_per_mM_s`
10. `topt_C`
11. `tm_C`

The `kcat/Km` field is preserved as a dataset-provided value, not recomputed
from displayed `kcat` and `Km`. Missing kinetic values appear as `nan`.

The primary sequence key is the KEGG gene entry
`organism_code:gene_id`, for example `aaa:Acav_0021`. UniProt accessions in the
GotEnzymes supplementary files are cross-references, not the primary sequence
keys.

## Data Pipeline

Inspect the archive:

```bash
python3 scripts/inspect_dataset.py --zip data/input_data.zip --sample-rows 8
```

Small KEGG remap smoke test:

```bash
python3 scripts/remap_kegg_sequences.py \
  --zip data/input_data.zip \
  --ec 1.1.1.3 \
  --limit 10 \
  --max-enzyme-files 5 \
  --out-fasta outputs/kegg_remap/ec_1_1_1_3.fasta \
  --out-index outputs/kegg_remap/ec_1_1_1_3.sequence_index.tsv \
  --out-metadata outputs/kegg_remap/ec_1_1_1_3.metadata.tsv \
  --out-row-map outputs/kegg_remap/ec_1_1_1_3.rows.tsv
```

For full remapping, use a local KEGG dump rather than KEGG REST:

```bash
python3 scripts/remap_kegg_sequences.py \
  --zip data/input_data.zip \
  --all \
  --kegg-root /path/to/downloaded/kegg \
  --fetch-missing none \
  --out-fasta outputs/kegg_remap/all_kegg_sequences.fasta \
  --out-index outputs/kegg_remap/all_kegg_sequence_index.tsv \
  --out-row-map outputs/kegg_remap/all_gotenzymes_rows.tsv
```

Build trimmed training MSAs from clustered members:

```bash
/home/florian/miniforge3/envs/msa_design/bin/python scripts/build_training_msas_from_clusters.py \
  --input-fasta /path/to/all_clustered_sequences.fasta \
  --members /path/to/good_msa_members.tsv \
  --out-root /mnt/backup4TB/MSA_DESIGN/training_msas_50_identity_core_gaptrim \
  --kalign tools/kalign/bin/kalign \
  --workers 8 \
  --kalign-threads 2 \
  --max-column-gap 0.20 \
  --max-sequence-gap 0.30 \
  --min-sequences 16 \
  --min-columns 50 \
  --min-residues 30 \
  --sequence-index /path/to/all_sequence_index.tsv.gz \
  --reaction-rows /path/to/all_reaction_parameters_uniprot.tsv.gz
```

Aggregate sequence-level labels, including the current `organism_code`
categorical field:

```bash
/home/florian/miniforge3/envs/msa_design/bin/python scripts/build_sequence_label_summary.py \
  --reaction-rows /mnt/backup4TB/MSA_DESIGN/training_msas_50_identity_core_gaptrim/kept_reaction_parameters.tsv.gz \
  --out /mnt/backup4TB/MSA_DESIGN/training_msas_50_identity_core_gaptrim/sequence_label_summary.tsv.gz
```

Precompute active ESM-MSA token-grid embeddings:

```bash
/home/florian/miniforge3/envs/msa_design/bin/python scripts/precompute_training_msa_embeddings.py \
  --msa-manifest /mnt/backup4TB/MSA_DESIGN/training_msas_50_identity_core_gaptrim/msa_manifest.tsv \
  --out-dir /mnt/backup4TB/MSA_DESIGN/training_msas_50_identity_core_gaptrim/esm_msa_token_embeddings_col \
  --weights weights/esm_msa1b_t12_100M_UR50S.pt \
  --device cuda \
  --max-seqs 64 \
  --max-cols 1023 \
  --dtype float16 \
  --store-token-embeddings
```

## Training

The generic launcher is `scripts/run_mean_start_ccdd_full_profile_row.sh`.
Override its environment variables for the active architecture:

```bash
DATA_ROOT=/media/florian/dc434ad3-38cd-442f-b7af-41802cfa5baa/MSA_DESIGN/training_msas_50_identity_core_gaptrim \
EMBED_DIR=/media/florian/dc434ad3-38cd-442f-b7af-41802cfa5baa/MSA_DESIGN/training_msas_50_identity_core_gaptrim/esm_msa_token_embeddings_col \
MEMORY_MODE=profile_msa_axial \
PROFILE_FEATURE_MODE=no_aa_frequency \
BATCH_SIZE=1 \
D_MODEL=192 \
CONTINUOUS_TARGET_MODE=target_row_embedding \
MASKED_ROWS_PER_MSA_MIN=4 \
MASKED_ROWS_PER_MSA_MAX=5 \
MSA_EMBEDDING_DTYPE=float16 \
AMP=fp16 \
MSA_AXIAL_LAYERS=1 \
CONSENSUS_LOSS_MODE=residual \
CONSENSUS_MATCH_WEIGHT=0.35 \
NONCONSENSUS_WEIGHT=2.5 \
UNOBSERVED_NONCONSENSUS_WEIGHT=1.0 \
MAX_SEQUENCE_LOSS_WEIGHT=3.0 \
CONDITION_MASK_PROB=0.0 \
VAL_BATCHES=64 \
CHECKPOINT_EVERY_STEPS=500 \
MAX_STEPS=544000 \
scripts/run_mean_start_ccdd_full_profile_row.sh \
  /media/florian/dc434ad3-38cd-442f-b7af-41802cfa5baa/MSA_DESIGN/training_msas_50_identity_core_gaptrim/mean_start_ccdd_current_run
```

To partial-resume when adding a condition token or head:

```bash
RESUME_CHECKPOINT=/path/to/mean_start_ccdd.best.pt \
ALLOW_PARTIAL_RESUME=1 \
RESET_OPTIMIZER=1 \
scripts/run_mean_start_ccdd_full_profile_row.sh /path/to/new_out_dir
```

Current mixed-MSA context launcher:

```bash
DATA_ROOT=/media/florian/dc434ad3-38cd-442f-b7af-41802cfa5baa/MSA_DESIGN/training_msas_50_identity_core_gaptrim \
scripts/run_mean_start_ccdd_mixed_msa64.sh
```

That wrapper resumes from the organism-code checkpoint, sets
`MEMORY_MODE=profile_msa_axial`, `MIXED_MSA_CONTEXT_ROWS=64`,
`MASKED_ROWS_PER_MSA_MIN=1`, `MASKED_ROWS_PER_MSA_MAX=1`, fp16 cached token
embeddings, fp16 AMP, and the residual objective.

Training outputs:

- `train.log`, launcher and trainer logs.
- `metrics.tsv`, train-window and validation metrics.
- `mean_start_ccdd.latest.pt`, latest checkpoint.
- `mean_start_ccdd.best.pt` plus `mean_start_ccdd.best.json`, best validation
  checkpoint by loss.
- `decode_step_*.fasta`, deterministic decode snapshots.

## Evaluation And Generation

Compare deterministic model decode against leave-one-row-out consensus:

```bash
/home/florian/miniforge3/envs/msa_design/bin/python scripts/evaluate_checkpoint_vs_consensus.py \
  --checkpoint /path/to/mean_start_ccdd.best.pt \
  --embedding-manifest /media/florian/dc434ad3-38cd-442f-b7af-41802cfa5baa/MSA_DESIGN/training_msas_50_identity_core_gaptrim/esm_msa_token_embeddings_col/embedding_manifest.tsv \
  --label-summary /media/florian/dc434ad3-38cd-442f-b7af-41802cfa5baa/MSA_DESIGN/training_msas_50_identity_core_gaptrim/sequence_label_summary.tsv.gz \
  --path-rewrite /mnt/backup4TB=/media/florian/dc434ad3-38cd-442f-b7af-41802cfa5baa \
  --split test \
  --example-limit 2048 \
  --batch-size 4 \
  --device cuda \
  --amp checkpoint \
  --summary-json /path/to/summary.json \
  --out-tsv /path/to/predictions.tsv
```

In this benchmark, "model beats consensus" means the decoded sequence has
higher identity to the hidden target row than the leave-one-row-out consensus
sequence for that same held-out row.

Generate one baseline-vs-thermostable contrast:

```bash
/home/florian/miniforge3/envs/msa_design/bin/python scripts/generate_thermostability_contrast.py \
  --checkpoint /path/to/mean_start_ccdd.best.pt \
  --embedding-manifest /media/florian/dc434ad3-38cd-442f-b7af-41802cfa5baa/MSA_DESIGN/training_msas_50_identity_core_gaptrim/esm_msa_token_embeddings_col/embedding_manifest.tsv \
  --label-summary /media/florian/dc434ad3-38cd-442f-b7af-41802cfa5baa/MSA_DESIGN/training_msas_50_identity_core_gaptrim/sequence_label_summary.tsv.gz \
  --sequence-manifest /media/florian/dc434ad3-38cd-442f-b7af-41802cfa5baa/MSA_DESIGN/training_msas_50_identity_core_gaptrim/sequence_manifest.tsv.gz \
  --path-rewrite /mnt/backup4TB=/media/florian/dc434ad3-38cd-442f-b7af-41802cfa5baa \
  --device cuda \
  --amp checkpoint \
  --msa-embedding-dtype float16 \
  --category-override organism_code=eco \
  --thermo-topt 68 \
  --thermo-tm 78 \
  --out-dir /path/to/thermostability_contrast
```

Generate a batch of thermostability contrasts:

```bash
/home/florian/miniforge3/envs/msa_design/bin/python scripts/generate_thermostability_batch.py \
  --checkpoint /path/to/mean_start_ccdd.best.pt \
  --embedding-manifest /media/florian/dc434ad3-38cd-442f-b7af-41802cfa5baa/MSA_DESIGN/training_msas_50_identity_core_gaptrim/esm_msa_token_embeddings_col/embedding_manifest.tsv \
  --label-summary /media/florian/dc434ad3-38cd-442f-b7af-41802cfa5baa/MSA_DESIGN/training_msas_50_identity_core_gaptrim/sequence_label_summary.tsv.gz \
  --sequence-manifest /media/florian/dc434ad3-38cd-442f-b7af-41802cfa5baa/MSA_DESIGN/training_msas_50_identity_core_gaptrim/sequence_manifest.tsv.gz \
  --path-rewrite /mnt/backup4TB=/media/florian/dc434ad3-38cd-442f-b7af-41802cfa5baa \
  --out-dir /path/to/thermostability_batch \
  --target-count 50 \
  --device cuda \
  --amp checkpoint \
  --msa-embedding-dtype float16 \
  --thermo-topt 60 \
  --thermo-tm 75
```

`generate_thermostability_batch.py` can optionally rebuild inference MSAs with
farthest-Hamming rows:

```bash
--inference-msa-row-selection farthest_hamming \
--farthest-msa-rows 64 \
--farthest-max-cols 1024 \
--farthest-embedding-out-dir /path/to/rebuilt_embeddings
```

After folding batch FASTAs with ColabFold, summarize structure-side signals:

```bash
/home/florian/miniforge3/envs/msa_design/bin/python scripts/analyze_thermostability_salt_bridge_batch.py \
  --metadata /path/to/batch_metadata.tsv \
  --fold-dir /path/to/colabfold_outputs \
  --out-dir /path/to/salt_bridge_analysis

/home/florian/miniforge3/envs/msa_design/bin/python scripts/analyze_thermostability_hydrophobic_packing_batch.py \
  --metadata /path/to/batch_metadata.tsv \
  --fold-dir /path/to/colabfold_outputs \
  --out-dir /path/to/hydrophobic_packing_analysis \
  --salt-bridge-summary /path/to/salt_bridge_analysis/salt_bridge_batch_summary.tsv
```

## Current Results Snapshot

The strongest current pre-mixing organism-code checkpoint is:

```text
/media/florian/dc434ad3-38cd-442f-b7af-41802cfa5baa/MSA_DESIGN/training_msas_50_identity_core_gaptrim/mean_start_ccdd_organismcode_condtokens_targetrow_latents_direct_axial_reads_sharedgrid_grouped_4to5_noaa_esmmsa_tokens_fullgrid_fp16amp_residual_condtokens_unmasked_partialresume_frombest173500_20260805_211457/mean_start_ccdd.best.pt
```

Checkpoint metadata:

- Best step: 321500.
- Validation loss: 1.21568.
- Validation token accuracy: 0.64468.
- Validation residue accuracy: 0.62212.
- Validation non-consensus residue accuracy: 0.54503.

Held-out `test` benchmark on 2,048 examples:

- Model deterministic t0 residue accuracy: 0.588761.
- Model mean sequence identity: 0.610307.
- Leave-one-row-out consensus residue accuracy: 0.739413.
- Consensus mean sequence identity: 0.737670.
- Model beat consensus on 875 of 2,048 examples.
- Model variable-nonconsensus-position accuracy: 0.540072.

The mixed-64 MSA context experiment was stopped after it degraded strongly:
best step 333500, validation loss 2.69714, token accuracy 0.22105, residue
accuracy 0.10963.

One E. coli-conditioned thermostability contrast for cluster `23854` and target
`cps:CPS_3483` used `organism_code=eco`, `Topt=68`, and `Tm=78`. The baseline
and thermostable generated sequences had identity 0.9655. Custom-A3M ColabFold
folds of consensus vs thermostable gave mean pLDDT 89.05 and 88.38, both pTM
0.83, with CA RMSD 0.782 A over 203 C-alpha atoms.

These are research diagnostics, not validated design claims.

## Legacy And Baseline Helpers

These scripts remain in the repo but are no longer the main architecture:

- `scripts/embed_msas.py`, pilot MSA embedding for small aligned FASTA/A3M
  files.
- `scripts/train_predictor.py`, frozen MSA predictor prototype for numeric
  targets.
- `scripts/train_sequence_decoder.py`, older fixed-length conditional diffusion
  decoder using compressed MSA depth memory.
- `scripts/train_aligned_column_decoder.py`, aligned-column decoder prototype.
- `scripts/decode_aligned_column_checkpoint.py`, legacy checkpoint decoding.
- `scripts/diagnose_attention_feature_importance.py`, attention diagnostics.
- `scripts/fetch_family_sequences.py`, legacy UniProt lookup helper. For
  source-faithful GotEnzymes work, prefer KEGG remapping and the UniProt-backed
  training MSA build path.

## Development Checks

Run the unit tests:

```bash
/home/florian/miniforge3/envs/msa_design/bin/python -m pytest tests
```

Compile the active model and script surfaces after broad edits:

```bash
/home/florian/miniforge3/envs/msa_design/bin/python -m py_compile \
  msa_design_model/model.py \
  msa_design_model/__init__.py \
  scripts/train_mean_start_ccdd_from_cached_msas.py \
  scripts/evaluate_checkpoint_vs_consensus.py \
  scripts/generate_thermostability_contrast.py \
  scripts/generate_thermostability_batch.py \
  scripts/analyze_thermostability_salt_bridge_batch.py \
  scripts/analyze_thermostability_hydrophobic_packing_batch.py
```

Use `bash -n` for launcher syntax:

```bash
bash -n scripts/run_mean_start_ccdd_full_profile_row.sh
bash -n scripts/run_mean_start_ccdd_mixed_msa64.sh
```
