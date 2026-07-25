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

- `scripts/build_figure_sourcedata.py` — rebuilds the Fig 2 t-SNE + score CSVs
  from the analysis outputs (edit the hard-coded input paths to your checkout).
- `scripts/compute_brsascore.py` — recomputes the BR-SAScore column
  (`pip install --user --break-system-packages --no-deps BRSAScore`; rdkit required).
- Materials Project references (formation-energy comparison, bulk modulus) are
  pulled live with [`scripts/mp_reference_data.py`](../../scripts/mp_reference_data.py).

## Not included here (too large for git)

- The **full generated MOF dataset** (structures + properties) and DFT trajectories
  live on Zenodo (dataset DOI `10.5281/zenodo.18452718`) and MPContribs
  (`MOFGen_2025`) — see [Data access](../../README.md#data-access).
- Fig 3 diffusion Td/solvent foreground and bulk modulus are regenerated with the
  MOFSimplify / MACE workflows (not committed).
