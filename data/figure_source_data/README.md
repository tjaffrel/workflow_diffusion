# Figure source data

Raw numerical data behind the plotted panels of the MOFGen paper, versioned with
the code so a tagged release (archived to Zenodo) is self-contained. Plain CSV,
UTF-8, tidy/long format. Group labels: **Experiments** / **LLM-generated** /
**Diffusion Model + LLM** / **QMOF reference**.

| File | Columns | Figure / content |
| --- | --- | --- |
| `Figure2a_tSNE_coordinates.csv` | `group, tsne_x, tsne_y` | Fig 2 — t-SNE of linker chemical space (26,318 points) |
| `Figure2bc_synthesizability_scores.csv.gz` | `group, metric, value` | Fig 2 — distributions of SCScore / SA Score / SMILES length / BR-SAScore (gzip; `gunzip` to read) |
| `Figure3ab_thermal_solvent_QMOF_reference.csv` | `MOF_name, thermal_decomposition_temp_C, solvent_removal_stability, dataset` | Fig 3a/b — QMOF reference (20,375 MOFs) |
| `Figure3d_formation_energy_MOFGen_DFT.csv` | `mof_id, formula, formation_energy_eV_per_atom, energy_eV, method, space_group` | Fig 3d — full MOFGen DFT set (7,783 MOFs: 3,935 r2SCAN-D4 + 3,848 MACE-OMat-D3) |

## Regenerating

Both scripts read their paths from the environment (no hard-coded paths):

```bash
export MOFGEN_SYNTH_SCORING_DIR=/path/to/diffusion/synth_scoring
export MOFGEN_SOURCE_DATA_OUT=$PWD          # optional; defaults to this folder
```

**`Figure2a_tSNE_coordinates.csv` and `Figure2bc_synthesizability_scores.csv.gz`:**
1. `python scripts/build_figure_sourcedata.py` — writes `Figure2a_tSNE_coordinates.csv`
   and `Figure2/Figure2bc_synthesizability_scores.csv` (SCScore, SA Score, SMILES length).
2. `python scripts/compute_brsascore.py` — writes `Figure2bc_BRSAScore.csv`
   (`pip install --user --break-system-packages --no-deps BRSAScore`; rdkit required).
3. Concatenate the BR-SAScore rows into the score CSV and gzip it:
   `tail -n +2 Figure2bc_BRSAScore.csv >> Figure2/Figure2bc_synthesizability_scores.csv && gzip -c Figure2/Figure2bc_synthesizability_scores.csv > Figure2bc_synthesizability_scores.csv.gz`

**`Figure3ab_thermal_solvent_QMOF_reference.csv`** — the QMOF MOFSimplify
thermal/solvent predictions (not produced by the scripts above; exported from the
MOFSimplify prediction run).

**`Figure3d_formation_energy_MOFGen_DFT.csv`** — pulled from MPContribs `MOFGen_2025`
with [`scripts/mp_data_extraction.py`](../../scripts/mp_data_extraction.py); the
Materials Project comparison distributions (formation energy, bulk modulus) come
live from [`scripts/mp_reference_data.py`](../../scripts/mp_reference_data.py).

(`build_figure_sourcedata.py` also emits small `Figure3/` diffusion subsets for
reference; those partial subsets are **not** the committed Fig 3 files above.)

## Not included here (too large for git)

- The **full generated MOF dataset** (structures + properties) and DFT trajectories
  live on Zenodo (dataset DOI `10.5281/zenodo.18452718`) and MPContribs
  (`MOFGen_2025`) — see [Data access](../../README.md#data-access).
- Fig 3 diffusion Td/solvent foreground and bulk modulus are regenerated with the
  MOFSimplify / MACE workflows (not committed).
