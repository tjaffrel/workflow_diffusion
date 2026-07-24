# Accessing the MOFGen data

This guide explains how to obtain every dataset behind the MOFGen paper's
figures, using only public sources and your own credentials. **No API keys or
tokens are stored in this repository** — you supply your own.

## 1. Materials Project reference data (no login beyond a free key)

The paper compares MOFGen-generated structures against Materials Project (MP)
reference distributions. These come from the **standard public MP API**, so any
user with a free key can reproduce them.

1. Get a free key at <https://next-gen.materialsproject.org/api#api-key>.
2. Make it available to the scripts:
   ```bash
   export MP_API_KEY="your_key_here"      # or put it in a local .env file
   ```
3. Pull the reference data:
   ```bash
   # Formation-energy reference (formation-energy comparison figure)
   python scripts/mp_reference_data.py formation-energy \
       --output data/mp_formation_energy.csv

   # Bulk-modulus reference (elastic-property figure)
   python scripts/mp_reference_data.py bulk-modulus \
       --output data/mp_bulk_modulus.csv
   ```
   Restrict to MOF-relevant chemistries with e.g. `--elements Zn,C,H,O`.

`.env` and any `*_key*` files are git-ignored; never commit your key.

## 2. The MOFGen_2025 dataset on MPContribs

The curated MOFGen_2025 contribution (computed MOF properties) is hosted on
MPContribs and pulled with the same `MP_API_KEY`:

```bash
python scripts/mp_data_extraction.py download --output data/mofgen_2025.csv
python scripts/mp_data_extraction.py query --metal Zr --output data/zr_mofs.csv
```

## 3. Full generated dataset and figure source data on Zenodo

For users who prefer a single download (no MP account needed at all):

| Content | Location |
| --- | --- |
| Full generated MOF dataset (structures + computed properties) | Zenodo dataset — DOI `10.5281/zenodo.18452718` |
| This code (release snapshot) | Zenodo software record for `tjaffrel/mofgen` |
| Per-figure **source data** (raw numbers behind each plot) | deposited with the article's Supplementary Information / Source Data |

## Which data underlies which figure

| Figure | Quantity | Source |
| --- | --- | --- |
| Linker chemical space | t-SNE coordinates | figure source data (CSV) |
| Synthesizability | SCScore, SA Score, SMILES length, BR-SAScore | figure source data (CSV); BR-SAScore via the `BRSAScore` package |
| Thermal / solvent stability | decomposition temperature, solvent-removal stability | r2SCAN MOF property set (this repo's workflow output) |
| Formation energy | eV/atom, MOFGen vs. MP | MOFGen: workflow output · MP: `scripts/mp_reference_data.py formation-energy` |
| Bulk modulus | GPa | Materials Project — `scripts/mp_reference_data.py bulk-modulus` |

> Bulk modulus is **not re-deposited**; it is retrieved directly from Materials
> Project so it always reflects the current MP release.
