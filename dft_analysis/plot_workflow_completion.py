"""Plot workflow step completion rates for the MOF screening pipeline.

Reproduces the workflow completion figure from arXiv:2504.14110.

Requires pre-computed data file (not included in repo):
- mof_wf_step_completion.json.gz

Usage:
    python dft_analysis/plot_workflow_completion.py
"""

from pathlib import Path

import plotly.graph_objects as go
from monty.serialization import loadfn
from seaborn import color_palette

ORDERED_STEPS = {
    "zeo++ input structure": "Zeo++ on<br>diffusion structure",
    "MACE relax": "MACE(-MP-0)<br>relaxation",
    "zeo++ mace-relaxed structure": "Zeo++ on<br>MACE-relaxed<br>structure",
    "gfn-xtb relax": "GFN1-xTB relaxation",
}


def main():
    data_file = Path("mof_wf_step_completion.json.gz")
    if not data_file.exists():
        print(
            f"Data file not found: {data_file}\n"
            "This script requires pre-computed workflow results not included in the repo."
        )
        return

    mof_completion = loadfn(data_file)

    data = [
        [ORDERED_STEPS[k], len(mof_completion[k])]
        for k in ORDERED_STEPS
    ]

    labels = [v[0] for v in data[1:]]
    pctgs = [100 * v[1] / data[0][1] for v in data[1:]]

    colors = [
        f"rgb({','.join(str(int(round(f * 255))) for f in v)})"
        for v in color_palette("colorblind")
    ]

    base_axis_opts = {
        "title_font_size": 32,
        "title_font_color": "black",
        "tickfont_color": "black",
        "tickfont_size": 28,
        "title_font_family": "Arial",
    }

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=labels,
            y=pctgs,
            marker_color=colors[0],
            text=[
                f"{v:.1f}%"
                + ("" if i > 0 else f"<br>of {data[0][1]}<br>structures")
                for i, v in enumerate(pctgs)
            ],
            textfont_size=32,
            textfont_family="Arial",
        )
    )

    h = 800
    fig.update_layout(
        height=h,
        width=int(h * 7 / 5),
        yaxis={
            "title_text": "Percentage of Structures (%)",
            "showline": True,
            "linecolor": "black",
            "linewidth": 2,
            **base_axis_opts,
        },
        xaxis={
            "showline": True,
            "linecolor": "black",
            "linewidth": 2,
            "tickangle": 0,
            **base_axis_opts,
        },
        margin={"t": 0, "r": 0, "b": 0, "l": 0},
        bargroupgap=0.1,
        bargap=0.0,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.write_image("workflow_completion.pdf", scale=3)
    print("Wrote workflow_completion.pdf")


if __name__ == "__main__":
    main()
