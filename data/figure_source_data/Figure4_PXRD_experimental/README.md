# Experimental PXRD — raw patterns (Figure 4 / synthesis)

Raw diffractometer output for the synthesized AI-MOFs (2θ vs intensity). Files
are kept in their original instrument formats.

| File | Format | Tentative AI-MOF (⚠️ confirm) |
| --- | --- | --- |
| `AA_3-12-2_Zn-DMF_pxrd.txt` | `Angle,Intensity` (CSV-like) | AI-MOF-4 (zinc dimethyl fumarate, DMF) |
| `Theo-Fumeric-1-DMF-Wet-2.ras` | Rigaku `.ras` (2θ intensity 1.0 per line) | AI-MOF-3 or 4 (zinc dimethyl fumarate) |
| `Al_L36_PXRD.dat` | GSAS FXYE | AI-MOF-5 (aluminum dimethyl fumarate) |
| `Perylene_Zn_MOF_PXRD.opju` | OriginLab project (binary) | AI-MOF-2 (perylene-based) |

**To confirm / still missing:** the mapping above is inferred from filenames and
should be verified by the authors. Raw PXRD for **AI-MOF-1** (zinc muconate),
**AI-MOF-6** and **AI-MOF-7** (triazole mixed-linker) is **not yet located**.

`.opju` is a proprietary OriginLab format; if possible, re-export it as a plain
two-column (2θ, intensity) `.txt`/`.csv` for a fully open record.
