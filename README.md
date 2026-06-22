# MSA_DESIGN

Small reproducible bootstrap for checking assumptions in the Zenodo enzyme dataset, fetching matching UniProt sequences, and building pilot multiple sequence alignments by EC family.

Large local artifacts are intentionally ignored by git:

- `data/input_data.zip` from Zenodo record 17376050
- `data/cache/uniprot/` UniProt REST JSON cache files
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

The environment includes Python 3.10, NumPy, CUDA PyTorch (`pytorch-cuda=12.4`), and `fair-esm`. On this machine it detects the RTX 3060 via CUDA.

## Commands

Inspect the archive structure and a small row sample:

```bash
python3 scripts/inspect_dataset.py --zip data/input_data.zip --max-enzyme-files 5 --sample-rows 8
```

Run a fuller inspection by omitting `--max-enzyme-files`:

```bash
python3 scripts/inspect_dataset.py --zip data/input_data.zip
```

Fetch a small EC family from early organism files:

```bash
python3 scripts/fetch_family_sequences.py \
  --zip data/input_data.zip \
  --ec 1.1.1.3 \
  --limit 5 \
  --max-enzyme-files 10 \
  --out-fasta outputs/pilot_msas/ec_1_1_1_3.fasta \
  --out-metadata outputs/pilot_msas/ec_1_1_1_3.metadata.tsv
```

Build an MSA:

```bash
python3 scripts/build_msa.py \
  outputs/pilot_msas/ec_1_1_1_3.fasta \
  outputs/pilot_msas/ec_1_1_1_3.msa.fasta
```

Create a few pilot families end to end:

```bash
python3 scripts/pilot_families.py --families 3 --seqs-per-family 5 --scan-files 25
```

`build_msa.py` uses external `mafft` when available. If `mafft` is not on `PATH`, it falls back to a deterministic pure-Python center-star Needleman-Wunsch alignment. That fallback is only meant for tiny pilot MSAs and is not production-quality.

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

## UniProt Fetching

`fetch_family_sequences.py` first queries UniProt by exact KEGG cross-reference, for example `xref:KEGG-aaa\:Acav_0021`. If that does not return a sequence, it falls back to `gene_exact:<gene_id>`. Metadata TSV output records `kegg_xref_verified=True` only when the selected UniProt record contains the expected KEGG cross-reference such as `aaa:Acav_0021`.

The fetcher sleeps between REST calls by default and caches JSON responses under `data/cache/uniprot/`.

## Next Steps

- Cross-check the web-table column labels against primary dataset documentation if exact provenance matters for publication.
- Decide whether sequence families should be grouped by exact EC, EC prefix, reaction, substrate, organism domain, or combinations of those fields.
- Require verified KEGG cross-references for production sequence sets, or manually audit unverified fallbacks.
- Replace the fallback aligner with MAFFT or another production aligner for real training data.
- Scale MSA Transformer embedding beyond smoke tests and train/evaluate the predictor with proper family-level splits.
- Define train/validation splits by homology or family, not random rows, to avoid leakage in downstream design models.
