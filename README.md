# MSA_DESIGN

Small reproducible bootstrap for checking assumptions in the Zenodo enzyme dataset, fetching matching UniProt sequences, and building pilot multiple sequence alignments by EC family.

Large local artifacts are intentionally ignored by git:

- `data/input_data.zip` from Zenodo record 17376050
- `data/cache/uniprot/` UniProt REST JSON cache files
- `outputs/pilot_msas/` FASTA, metadata TSV, and MSA pilot outputs
- `weights/esm_msa1b_t12_100M_UR50S.pt`, ESM MSA Transformer weights

## Data Framing

This repository is for the first assumption-checking step of a sequence-design idea: encode MSAs with MSA Transformer, compress or project latent tensors, condition on enzyme metadata such as substrate, EC, and experimentally validated kinetic parameters when available, then decode or generate candidate sequence variants.

The current scripts do not assume the archive already contains `Km`, `kcat`, or any other named kinetic label. `input_data/enzymes/*.txt` rows have 11 tab-separated columns and no header in the archive. The scripts use conservative working names:

1. `gene_id`
2. `organism_code`
3. `domain`
4. `reaction_id`
5. `ec_numbers`
6. `compound_id`
7. `numeric_col_7_unlabeled`
8. `numeric_col_8_unlabeled`
9. `numeric_col_9_unlabeled`
10. `numeric_col_10_unlabeled`
11. `numeric_col_11_unlabeled`

Current evidence only: columns 7-9 look kinetic-like because they vary by compound/reaction and have missing values; columns 10-11 look like temperature and pH*10 ranges, but this is unconfirmed. Validate these meanings against the dataset paper, Zenodo documentation, or upstream generation code before using them as supervised labels.

Supplementary files in the zip provide headers for compounds, domains, EC names, gene cross-references, organisms, and reactions. The pipeline reads the zip directly and does not extract the full archive.

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

## UniProt Fetching

`fetch_family_sequences.py` first queries UniProt by exact KEGG cross-reference, for example `xref:KEGG-aaa\:Acav_0021`. If that does not return a sequence, it falls back to `gene_exact:<gene_id>`. Metadata TSV output records `kegg_xref_verified=True` only when the selected UniProt record contains the expected KEGG cross-reference such as `aaa:Acav_0021`.

The fetcher sleeps between REST calls by default and caches JSON responses under `data/cache/uniprot/`.

## Next Steps

- Confirm the unlabeled numeric columns against primary dataset documentation before treating them as `Km`, `kcat`, temperature, or pH.
- Decide whether sequence families should be grouped by exact EC, EC prefix, reaction, substrate, organism domain, or combinations of those fields.
- Require verified KEGG cross-references for production sequence sets, or manually audit unverified fallbacks.
- Replace the fallback aligner with MAFFT or another production aligner for real training data.
- Define train/validation splits by homology or family, not random rows, to avoid leakage in downstream design models.
