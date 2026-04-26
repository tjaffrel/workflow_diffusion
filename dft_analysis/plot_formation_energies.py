"""Plot formation energy distributions for MOFGen structures.

Reproduces Fig. S3(d) from arXiv:2504.14110.

Requires pre-computed data files (not included in repo):
- unary_energy_per_atom.json.gz  — elemental reference energies
- mof_final_structures.json.gz   — relaxed MOF structures with energies

Usage:
    python dft_analysis/plot_formation_energies.py
"""

from pathlib import Path

import numpy as np
import plotly.graph_objects as pgo
from monty.serialization import loadfn
from scipy.stats import gaussian_kde
from seaborn import color_palette


def comp_from_struct_dict(struct_dict):
    """Extract composition dict from a pymatgen structure dict."""
    from pymatgen.core import Structure

    struct = Structure.from_dict(struct_dict)
    return {str(el): amt for el, amt in struct.composition.as_dict().items()}


def group_by_metal(final_structures):
    """Group formation energies by metal node."""
    by_comp = {}
    for entry in final_structures.values():
        comp = comp_from_struct_dict(entry["structure"])
        metals = sorted(
            el for el in comp if el not in ("C", "H", "N", "O", "S", "F", "Cl")
        )
        node_key = "-".join(metals)
        by_comp.setdefault(node_key, []).append(entry["e_form_per_atom"])
    return by_comp


def plot_formation_energies(by_comp, mp_e_forms=None, show_bins=False):
    """Create formation energy distribution plots."""
    palette = [
        f"rgb({','.join(str(int(round(255 * f))) for f in v)})"
        for v in color_palette("bright")
    ]

    node_keys = sorted(
        [k for k in by_comp if k in ("Zn", "Mg", "Al")],
        key=lambda k: len(by_comp.get(k, [])),
        reverse=True,
    )
    non_alpha_colors = {k: palette[i] for i, k in enumerate(node_keys)}

    if mp_e_forms is not None:
        by_comp["Materials Project"] = mp_e_forms
        node_keys_mp = ["Zn", "Materials Project"]
        non_alpha_colors["Materials Project"] = palette[len(node_keys)]

    bds = (
        min(min(v) for k, v in by_comp.items() if k != "Materials Project"),
        max(max(v) for k, v in by_comp.items() if k != "Materials Project"),
    )

    base_axis_opts = {
        "title_font_size": 32,
        "title_font_color": "black",
        "title_font_family": "Arial",
        "tickfont_color": "black",
        "tickfont_size": 28,
        "showline": True,
        "linewidth": 2,
        "linecolor": "black",
    }

    nbin = 25
    configs = [(node_keys, bds, "mof_formation_energies")]
    if mp_e_forms is not None:
        configs.append((node_keys_mp, (-2.0, 0.5), "zn_mof_formation_energies"))

    for sorted_nodes, plot_bds, base_name in configs:
        for show_bins_flag in (False, True) if show_bins else (False,):
            suffix = "_w_bins" if show_bins_flag else ""
            fig = pgo.Figure()

            counts, bins = {}, {}
            for node in sorted_nodes:
                is_mp = node == "Materials Project"
                counts[node], bins[node] = np.histogram(
                    by_comp[node],
                    bins=200 if is_mp else nbin,
                    range=(-2.0, 0.5) if is_mp else bds,
                    density=True,
                )

            for node in sorted_nodes:
                gauss_kde = gaussian_kde(by_comp[node])
                interp = np.linspace(
                    (1.0 - 0.1 * np.sign(plot_bds[0])) * plot_bds[0],
                    (1.0 + 0.1 * np.sign(plot_bds[1])) * plot_bds[1],
                    2000,
                )
                fig.add_trace(
                    pgo.Scatter(
                        x=interp,
                        y=gauss_kde(interp),
                        marker={"color": non_alpha_colors[node]},
                        name=node,
                        mode="lines",
                        fill=None if show_bins_flag else "tozeroy",
                    )
                )

            h = 800
            fig.update_layout(
                barmode="overlay",
                height=h,
                width=int(h * 7 / 5),
                yaxis={"title_text": "Density", **base_axis_opts},
                xaxis={
                    "title_text": "Formation energy (eV/atom)",
                    "range": [1.2 * b for b in plot_bds],
                    **base_axis_opts,
                },
                legend={
                    "bgcolor": "rgba(0,0,0,0)",
                    "font_size": 28,
                    "font_color": "black",
                },
                margin={"t": 0, "r": 0, "b": 0, "l": 0},
                plot_bgcolor="white",
                paper_bgcolor="white",
            )
            fig.write_image(f"{base_name}{suffix}.pdf", scale=4)
            print(f"Wrote {base_name}{suffix}.pdf")


def main():
    data_dir = Path(".")

    structures_file = data_dir / "mof_final_structures.json.gz"
    if not structures_file.exists():
        print(
            f"Data file not found: {structures_file}\n"
            "This script requires pre-computed DFT results not included in the repo."
        )
        return

    final_structures = loadfn(structures_file)
    by_comp = group_by_metal(final_structures)

    print("Metal nodes found:")
    for node, e_forms in sorted(by_comp.items()):
        print(f"  {node}: {len(e_forms)} structures")

    # Optionally load Materials Project reference data
    mp_e_forms = None
    try:
        from mp_api.client import MPRester

        with MPRester() as mpr:
            thermo_docs = mpr.materials.thermo.search(
                thermo_types=["R2SCAN"],
                fields=["formation_energy_per_atom"],
            )
        mp_e_forms = [d.formation_energy_per_atom for d in thermo_docs]
        print(f"  Materials Project: {len(mp_e_forms)} entries")
    except Exception as e:
        print(f"Skipping Materials Project comparison: {e}")

    plot_formation_energies(by_comp, mp_e_forms)


if __name__ == "__main__":
    main()
