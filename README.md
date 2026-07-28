# MSA_DESIGN

Small reproducible bootstrap for checking assumptions in the Zenodo enzyme dataset, remapping GotEnzymes rows to their source KEGG protein sequences, and building pilot multiple sequence alignments by EC family.

Large local artifacts are intentionally ignored by git:

- `data/input_data.zip` from Zenodo record 17376050
- `data/cache/kegg_aaseq/` KEGG REST amino-acid FASTA cache files
- `data/cache/uniprot/` legacy UniProt REST JSON cache files
- `outputs/pilot_msas/` FASTA, metadata TSV, and MSA pilot outputs
- `outputs/embeddings/` MSA Transformer `.npz` tensors and metadata JSON files
- `weights/esm_msa1b_t12_100M_UR50S.pt`, ESM MSA Transformer weights
- `weights/esm_msa1b_t12_100M_UR50S-contact-regression.pt`, tiny `fair-esm` sidecar expected by local model loading

## Data Framing

This repository is for the first assumption-checking step of a sequence-design idea: encode MSAs with MSA Transformer, compress or project latent tensors, condition on enzyme metadata such as substrate, EC, and experimentally validated kinetic parameters when available, then decode or generate candidate sequence variants.

`input_data/enzymes/*.txt` rows have 11 tab-separated columns and no header in the archive. The matching web-table header identifies the schema as:

1. `gene_id`
2. `organism_code`
3. `domain`
4. `reaction_id`
5. `ec_numbers`
6. `compound_id`
7. `kcat_1_per_s` (`kcat[1/s]`)
8. `km_mM` (`Km[mM]`)
9. `kcat_over_km_1_per_mM_s` (`kcat/Km[1/mM-s]`)
10. `topt_C` (`Topt[°C]`)
11. `tm_C` (`Tm[°C]`)

The `kcat/Km` column is preserved as a dataset-provided value, not recomputed from the displayed `kcat` and `Km` fields. Missing kinetic values appear as `nan` in some rows.

Supplementary files in the zip provide headers for compounds, domains, EC names, gene cross-references, organisms, and reactions. The pipeline reads the zip directly and does not extract the full archive.

GotEnzymes mined its prediction inputs from KEGG, including per-organism protein sequences, compound structures, and EC/reaction associations. The correct primary sequence identifier for an enzyme row is therefore the KEGG gene entry `organism_code:gene_id`, for example `aaa:Acav_0021`. UniProt accessions in `input_data/supplementary/gene.txt` are cross-references, not the primary sequence key for this dataset.


## Dedicated Conda Environment

A separate CUDA-enabled conda environment is defined in `environment.yml` and has been created locally as `msa_design` under Miniforge:

```bash
/home/florian/miniforge3/bin/mamba env create -f environment.yml
conda activate msa_design
```

Without shell activation, run commands through the environment directly:

```bash
/home/florian/miniforge3/envs/msa_design/bin/python scripts/embed_msas.py --help
```

The environment includes Python 3.10, NumPy, CUDA PyTorch (`pytorch-cuda=12.4`), Kalign for fast protein alignment, and `fair-esm`. On this machine it detects the RTX 3060 via CUDA.

## Commands

Inspect the archive structure and a small row sample:

```bash
python3 scripts/inspect_dataset.py --zip data/input_data.zip --max-enzyme-files 5 --sample-rows 8
```

Run a fuller inspection by omitting `--max-enzyme-files`:

```bash
python3 scripts/inspect_dataset.py --zip data/input_data.zip
```

Fetch/remap a small EC family from early organism files:

```bash
python3 scripts/remap_kegg_sequences.py \
  --zip data/input_data.zip \
  --ec 1.1.1.3 \
  --limit 5 \
  --max-enzyme-files 10 \
  --out-fasta outputs/pilot_msas/ec_1_1_1_3.fasta \
  --out-index outputs/pilot_msas/ec_1_1_1_3.sequence_index.tsv \
  --out-metadata outputs/pilot_msas/ec_1_1_1_3.metadata.tsv
```

Build an MSA:

```bash
/home/florian/miniforge3/envs/msa_design/bin/python scripts/build_msa.py \
  outputs/pilot_msas/ec_1_1_1_3.fasta \
  outputs/pilot_msas/ec_1_1_1_3.msa.fasta
```

Create a few pilot families end to end:

```bash
python3 scripts/pilot_families.py --families 3 --seqs-per-family 5 --scan-files 25
```

`build_msa.py` uses external Kalign in `fast` mode by default when it is available, then MAFFT if Kalign is missing. If neither is available, it falls back to a deterministic pure-Python center-star Needleman-Wunsch alignment. That fallback is only meant for tiny pilot MSAs and is not production-quality.

## KEGG Sequence Remapping

`scripts/remap_kegg_sequences.py` writes normalized sequence/remap artifacts from GotEnzymes rows:

- FASTA: one source KEGG protein sequence per selected unique `organism_code:gene_id`
- sequence index TSV: one row per selected KEGG gene entry, with sequence status/length/source
- per-gene metadata TSV: one row per selected KEGG gene entry, aggregating all selected reaction/compound/property rows
- optional row-map TSV: one row per original GotEnzymes property row, with `sequence_id` pointing to the matching KEGG FASTA record

For small selections, the script can fetch exact source sequences through KEGG REST:

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

For the full local archive, use a licensed local KEGG dump rather than KEGG REST:

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

The local archive currently contains 59,631,387 GotEnzymes property rows over 8,368,096 unique KEGG gene entries. KEGG REST `get ... /aaseq` is limited to 10 entries per request, so a complete remap should use the local KEGG `genes/organisms/<org>/<org>.pep` files that GotEnzymes was originally built from. The REST path is for smoke tests and small EC/organism subsets.

For a balanced working subset, `scripts/sample_kegg_family_sequences.py` scans the GotEnzymes archive, samples EC/reaction families with stable hash sampling, fetches the exact KEGG source sequences, and writes combined plus per-family FASTAs:

```bash
/home/florian/miniforge3/envs/msa_design/bin/python scripts/sample_kegg_family_sequences.py \
  --target-sequences 5000 \
  --seqs-per-family 50 \
  --min-seqs-per-family 50 \
  --out-dir outputs/kegg_representative_5000 \
  --sleep-seconds 0.4 \
  --max-rest-requests 1000
```

The local `outputs/kegg_representative_5000/` run contains:

- `combined.fasta`: 5,000 exact KEGG protein sequences
- `families.tsv`: 100 selected EC/reaction families, balanced across EC classes
- `entries.tsv`: selected KEGG entries with family assignment, sequence metadata, and per-entry GotEnzymes kinetic values (`kcat_1_per_s_values`, `km_mM_values`, `kcat_over_km_1_per_mM_s_values`, `topt_C_values`, `tm_C_values`)
- `sequence_index.tsv`: sequence status, length, source cache path, and KEGG FASTA header
- `families/*.fasta`: one 50-sequence FASTA per selected family
- `msas/*.msa.fasta`: Kalign alignments for all selected families
- `topup_replacements.tsv`: 10 selected KEGG entries that lacked `aaseq` responses and their replacement entries

## MSA Transformer Embedding

`scripts/embed_msas.py` embeds aligned FASTA/A3M files with ESM MSA Transformer using the local ignored weights at `weights/esm_msa1b_t12_100M_UR50S.pt`. The real embedding path requires an environment with `torch`, `numpy`, and `esm`/`fair-esm` installed; use the dedicated `msa_design` environment above. The script imports `torch` and `esm` lazily, so parsing and shape checks can run without those packages.

Dry-run one pilot MSA without loading the model:

```bash
/home/florian/miniforge3/envs/msa_design/bin/python scripts/embed_msas.py \
  --msa outputs/pilot_msas/ec_1_1_1_3.msa.fasta \
  --out-dir outputs/embeddings \
  --dry-run
```

Dry-run every pilot MSA:

```bash
/home/florian/miniforge3/envs/msa_design/bin/python scripts/embed_msas.py \
  --msa-glob 'outputs/pilot_msas/*.msa.fasta' \
  --out-dir outputs/embeddings \
  --dry-run
```

Prepare reproducible embedding commands or a TSV manifest:

```bash
/home/florian/miniforge3/envs/msa_design/bin/python scripts/prepare_embedding_jobs.py --dry-run
/home/florian/miniforge3/envs/msa_design/bin/python scripts/prepare_embedding_jobs.py --out-manifest outputs/embeddings/jobs.tsv
```

Run real embedding in the dedicated environment:

```bash
/home/florian/miniforge3/envs/msa_design/bin/python scripts/embed_msas.py \
  --msa-glob 'outputs/pilot_msas/*.msa.fasta' \
  --weights weights/esm_msa1b_t12_100M_UR50S.pt \
  --out-dir outputs/embeddings \
  --device auto \
  --max-seqs 64 \
  --max-cols 1024
```

Each MSA writes `outputs/embeddings/<stem>.npz` plus `outputs/embeddings/<stem>.metadata.json`. The NPZ contains:

- `token_embeddings`, shape `rows x cols x hidden_dim`, unless `--pool-only` or `--no-include-token-embeddings` is used
- `row_embeddings`, mean embedding per MSA row over non-gap amino-acid positions
- `col_embeddings`, mean embedding per aligned column over non-gap amino-acid positions
- `query_embedding`, row-0 mean over non-gap positions
- `msa_embedding`, global mean over all non-gap positions
- `aa_mask`, `gap_mask`, `row_aa_counts`, and `col_aa_counts`

A3M lowercase insertion characters and dots are removed before validation. Gaps (`-`) are preserved. MSAs are deterministically cropped to the first `--max-seqs` rows and first `--max-cols` columns, with crop notes recorded in metadata. The next modeling step should consume the pooled arrays or project/compress `token_embeddings` before conditioning on enzyme metadata.


## Trainable Predictor Prototype

The first trainable stack lives in `msa_design_model/` and keeps the MSA Transformer boundary frozen:

- `FrozenMSATransformerEncoder` wraps local `fair-esm` MSA Transformer loading, sets all MSAformer parameters to `requires_grad=False`, and exposes an `encode()` method for future end-to-end use.
- `RowColumnProjector` consumes frozen token embeddings with shape `B x R x L x H`, builds masked row and column context features, fuses `token + row + column`, and projects/pools to an `L x d` matrix per MSA.
- `NumericConditionTokenBank` has separate numeric embedding heads for `kcat_1_per_s`, `km_mM`, `kcat_over_km_1_per_mM_s`, `topt_C`, and `tm_C`. Missing values use learned missing tokens.
- `EnzymeMSAPredictor` appends the numeric condition tokens to the projected `L x d` MSA matrix, runs a small trainable Transformer encoder, pools, and predicts a requested numeric target.

Smoke-train the predictor from the precomputed EC-family embeddings:

```bash
/home/florian/miniforge3/envs/msa_design/bin/python scripts/train_predictor.py \
  --epochs 3 \
  --batch-size 2 \
  --d-model 64 \
  --layers 1 \
  --heads 4 \
  --device cpu \
  --out-checkpoint outputs/checkpoints/predictor_smoke.pt
```

By default, `scripts/train_predictor.py` predicts `kcat_1_per_s` using `outputs/embeddings/ec_*.npz` and matching `outputs/pilot_msas/<stem>.metadata.tsv` rows. Semicolon-separated numeric metadata values are aggregated as the mean of finite values. The target field still has a condition-token slot, but it is masked as missing unless `--include-target-as-condition` is explicitly passed, avoiding target leakage in the default setup.

The saved checkpoint contains only trainable predictor weights and config; MSA Transformer weights remain frozen/precomputed.

## Sequence Diffusion Decoder Prototype

The sequence-design decoder keeps the MSA Transformer boundary frozen, then adds two trainable stages:

- `MSADepthScaler` consumes frozen token embeddings with shape `B x R x L x H` and compresses the MSA row/depth axis into `B x L x d` with learned depth attention. The aligned-column length `L` can differ per family and is carried as a mask.
- `MetadataConditionTokenBank` encodes enzyme metadata as prepended context tokens. Numeric fields such as `kcat_1_per_s`, `km_mM`, `kcat_over_km_1_per_mM_s`, `topt_C`, and `tm_C` pass through per-field MLPs after dataset normalization. Categorical fields such as `ec_numbers`, `reaction_ids`, and `compound_ids` pass through learned vocab embeddings and support semicolon-separated multi-value metadata.
- Condition tokens run through a small self-attention mixer, then they are prepended to the MSA latent memory. They are not generated as sequence tokens.
- `SequenceDiffusionDecoder` generates a fixed `1280` output positions by default. It cross-attends from those fixed output positions into the variable-length `B x L x d` MSA latent.
- `MSASequenceDiffusionModel` combines both pieces for training or sampling.

Protein sequences use `*` as the STOP token. Targets are encoded as residues, then the first `*`, then repeated trailing `*` tokens until the fixed decoder length. Residues and the first `*` get full token-loss weight; trailing padding `*` tokens get a low weight so the decoder learns termination without being dominated by padding. The training loss reconstructs the protein sequence under the supplied condition prefix, rather than predicting metadata tokens as part of the protein alphabet. Training also logs weighted token reconstruction accuracy as a metric; the optimized objective stays differentiable.

Decoder-side starts default to `--decoder-start-mode mean`: every output position starts from the learned mean amino-acid embedding, with cross-attention into the remaining MSA rows and metadata condition tokens. This is the default for new continuous reconstruction runs because q-sampled target embeddings proved too easy to denoise. Sequence-only discrete diffusion is available with `--decoder-start-mode discrete_mask`: at timestep `t`, target token IDs are replaced by a real `<MASK>` token with probability `(t + 1) / diffusion_timesteps`. By default the CE loss is applied only to corrupted positions, and `--condition-dropout` randomly replaces MSA/metadata memory with a learned null token so the discrete sampler can use classifier-free guidance. Sampling starts from all `<MASK>` tokens and iteratively fills/remasks by confidence.

`scripts/train_sequence_decoder.py` creates one training example per aligned MSA row when matching metadata TSV rows are available. The same family MSA can therefore condition on different row-level kinetic/reaction metadata and reconstruct the corresponding row sequence.
Requested numeric condition fields must contain at least one observed value by default; this prevents accidental all-missing kinetic tokens. If a missing-only ablation is intentional, pass `--allow-empty-numeric-condition-fields`.

If a decoder metadata directory was built before kinetic value columns were propagated, backfill it directly from the GotEnzymes archive:

```bash
/home/florian/miniforge3/envs/msa_design/bin/python scripts/enrich_metadata_from_gotenzymes.py \
  --metadata-dir outputs/training/okay24_20260713_233827/metadata \
  --in-place
```

Smoke-train the conditioned decoder from precomputed MSA embeddings and metadata:

```bash
/home/florian/miniforge3/envs/msa_design/bin/python scripts/train_sequence_decoder.py \
  --epochs 1 \
  --batch-size 1 \
  --d-model 64 \
  --layers 1 \
  --heads 4 \
  --device cpu \
  --max-examples 2 \
  --max-sequence-length 1280 \
  --decoder-start-mode mean \
  --numeric-condition-fields kcat_1_per_s,km_mM,kcat_over_km_1_per_mM_s,topt_C,tm_C \
  --categorical-condition-fields ec_numbers,reaction_ids,compound_ids \
  --out-checkpoint outputs/checkpoints/sequence_diffusion_smoke.pt
```

For faster CPU wiring checks, use a smaller temporary `--max-sequence-length`; production runs should use the default `1280`.

## Legacy UniProt Fetching

`fetch_family_sequences.py` is kept as a legacy pilot helper. It queries UniProt by exact KEGG cross-reference, for example `xref:KEGG-aaa\:Acav_0021`, then falls back to `gene_exact:<gene_id>`. That is useful for ad-hoc cross-checking, but the source-faithful GotEnzymes remap should use `remap_kegg_sequences.py`.

The fetcher sleeps between REST calls by default and caches JSON responses under `data/cache/uniprot/`.

## Next Steps

- Decide whether sequence families should be grouped by exact EC, EC prefix, reaction, substrate, organism domain, or combinations of those fields.
- Cluster fetched enzymes into tight homolog groups before MSA embedding, then align each close group with Kalign. Broad EC-level MSAs are useful for smoke tests, but tight local alignments should be cleaner conditioning targets for the decoder.
- Scale MSA Transformer embedding beyond smoke tests and train/evaluate the predictor with proper family-level splits.
- Define train/validation splits by homology or family, not random rows, to avoid leakage in downstream design models.
