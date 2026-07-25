"""Rebuild the Figure 2 t-SNE coordinates and the SCScore/SA Score/SMILES-length
score CSV (and, for reference, the small Fig 3 diffusion subsets) from the
diffusion `synth_scoring` analysis outputs.

Paths are taken from the environment so the script is portable:
  MOFGEN_SYNTH_SCORING_DIR  input dir (your `diffusion/synth_scoring` checkout)
  MOFGEN_SOURCE_DATA_OUT    output dir (defaults to this repo's figure_source_data/)
"""
import json, gzip, csv, re, os
from pathlib import Path

DIFF = Path(os.environ.get("MOFGEN_SYNTH_SCORING_DIR", "")).expanduser()
OUT  = Path(os.environ.get("MOFGEN_SOURCE_DATA_OUT",
                           Path(__file__).resolve().parent.parent)).expanduser()
if not DIFF.is_dir():
    raise SystemExit(
        "Set MOFGEN_SYNTH_SCORING_DIR to your diffusion/synth_scoring checkout, e.g.\n"
        "  export MOFGEN_SYNTH_SCORING_DIR=/path/to/diffusion/synth_scoring")
(OUT/"Figure2").mkdir(parents=True, exist_ok=True)
(OUT/"Figure3").mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- Fig 2a: t-SNE
# Coordinates are embedded in tsne_mace_omat_diffusion_v3.svg (same approach as
# tsne_replot.py). Extract pixel positions per colour group, calibrate px->data
# via the axis tick labels, write (group, tsne_x, tsne_y).
svg = (DIFF/"tsne_mace_omat_diffusion_v3.svg").read_text()
GROUPS = {"#0173b2":"Experiments", "#029e73":"LLM-generated", "#de8f05":"Diffusion Model + LLM"}
pts = {g:[] for g in GROUPS.values()}
for pc in re.findall(r'<g id="PathCollection_\d+">.*?</g>', svg, re.DOTALL):
    uses = re.findall(r'<use[^>]*x="([\d.]+)"\s+y="([\d.]+)"[^>]*fill:\s*(#[0-9a-fA-F]{6})', pc)
    if len(uses) <= 1:
        continue
    for px, py, hexc in uses:
        hexc = hexc.lower()
        if hexc in GROUPS:
            pts[GROUPS[hexc]].append((float(px), float(py)))

def linfit(xs, ys):                       # least squares, no numpy
    n=len(xs)
    if n < 2:
        raise ValueError("need >=2 calibration ticks to fit, got %d" % n)
    sx=sum(xs); sy=sum(ys); sxx=sum(x*x for x in xs); sxy=sum(x*y for x,y in zip(xs,ys))
    denom = n*sxx - sx*sx
    if denom == 0:
        raise ValueError("degenerate calibration: all tick pixel positions identical")
    a=(n*sxy - sx*sy)/denom; b=(sy - a*sx)/n; return a,b

def calib(axis, idx):
    vals, pix = [], []
    for blk in re.findall(rf'<g id="{axis}_\d+">.*?(?=<g id="{axis}_\d+">|</g>\s*<g id="{axis[0]}axis)', svg, re.DOTALL):
        cm = re.search(r'<!--\s*([−\-]?\d+)\s*-->', blk)
        tr = re.search(r'translate\(([\d.]+)\s+([\d.]+)\)', blk)
        if cm and tr:
            vals.append(int(cm.group(1).replace("−","-"))); pix.append(float(tr.group(1+idx)))
    return linfit(pix, vals)

ax_a, ax_b = calib("xtick", 0)
ay_a, ay_b = calib("ytick", 1)
with open(OUT/"Figure2"/"Figure2a_tSNE_coordinates.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["group","tsne_x","tsne_y"])
    for g,lst in pts.items():
        for px,py in lst:
            w.writerow([g, round(ax_a*px+ax_b,4), round(ay_a*py+ay_b,4)])
n_tsne = sum(len(v) for v in pts.values())

# ------------------------------------------------ Fig 2b/c: SCS / SA / length
with open(DIFF/"reviewer_figures"/"cached_scores.json") as f: cache = json.load(f)
GMAP = {"exp":"Experiments", "llm":"LLM-generated", "diff":"Diffusion Model + LLM"}
METRICS = {"scs":"SCScore", "sa":"SA Score", "len":"SMILES length"}
rows=0
with open(OUT/"Figure2"/"Figure2bc_synthesizability_scores.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["group","metric","value"])
    for gk,gname in GMAP.items():
        for mk,mname in METRICS.items():
            for v in cache.get(f"{gk}_{mk}", []):
                w.writerow([gname, mname, v]); rows+=1

# ------------------------------------------------------- Fig 3: Td + solvent
with open(DIFF/"database_mof"/"high_priority_mofs_r2scan_d4_Td_solvent.json") as f:
    td = json.load(f)
with open(OUT/"Figure3"/"Figure3_thermal_decomposition.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["MOF_name","thermal_decomposition_temp_C","model_prediction_confidence"])
    for e in td:
        w.writerow([e.get("MOF_name"), e.get("Thermal Decomposition Temp"), e.get("Prediction")])

# ---------------------------------------------- Fig 3: formation energy (diffusion)
with gzip.open(DIFF/"database_mof"/"high_priority_mofs_r2scan_d4_relaxed.json.gz") as f: rel = json.load(f)
with open(OUT/"Figure3"/"Figure3_formation_energy.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["mof_id","formation_energy_eV_per_atom","dataset"])
    for k,v in rel.items():
        w.writerow([k, v.get("formation_energy_eV_per_atom"), "Diffusion Model + LLM (r2SCAN)"])

print("t-SNE points:", n_tsne, {g:len(v) for g,v in pts.items()})
print("score rows:", rows)
print("Td MOFs:", len(td), "| formation-energy MOFs:", len(rel))
print("x-calib a,b =", round(ax_a,4), round(ax_b,2), "| y-calib a,b =", round(ay_a,4), round(ay_b,2))
