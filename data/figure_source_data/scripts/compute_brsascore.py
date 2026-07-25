"""Compute BR-SAScore for the 3 linker sources (same inputs as reviewer_br_sascore.py)
and write a long-format CSV (group, metric, value) with metric 'BR-SAScore'.

Paths come from the environment for portability:
  MOFGEN_SYNTH_SCORING_DIR  input dir (your `diffusion/synth_scoring` checkout)
  MOFGEN_SOURCE_DATA_OUT    output dir (defaults to this repo's figure_source_data/)
The BR-SAScore rows are written to Figure2bc_BRSAScore.csv; concatenate them into
Figure2bc_synthesizability_scores.csv (see the folder README) to reproduce the
committed gzipped file."""
import os, csv, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
from BRSAScore import SAScorer

DIFF = Path(os.environ.get("MOFGEN_SYNTH_SCORING_DIR", "")).expanduser()
OUT  = Path(os.environ.get("MOFGEN_SOURCE_DATA_OUT",
                           Path(__file__).resolve().parent.parent)).expanduser() / "Figure2bc_BRSAScore.csv"
if not DIFF.is_dir():
    raise SystemExit(
        "Set MOFGEN_SYNTH_SCORING_DIR to your diffusion/synth_scoring checkout, e.g.\n"
        "  export MOFGEN_SYNTH_SCORING_DIR=/path/to/diffusion/synth_scoring")
OUT.parent.mkdir(parents=True, exist_ok=True)

def exp_smiles():
    with open(DIFF / "structure_10143_processed" / "linkers_df.csv", newline="") as f:
        for row in csv.DictReader(f):
            s = row["linker_canonical_smile"].strip()
            if s:
                yield s
def llm_smiles():
    with open(DIFF/"database_linker"/"all_SMILES.csv", newline="") as f:
        for row in csv.DictReader(f):
            s=row["SMILES"].strip()
            if s: yield s
def diff_smiles():
    seen=set()
    with open(DIFF/"database_linker"/"novel_smiles_validated.csv", newline="") as f:
        for row in csv.DictReader(f):
            s=(row.get("Standardized_SMILES") or row.get("Original_SMILES","")).strip()
            if s and s not in seen:
                seen.add(s); yield s

SRC=[("Experiments",exp_smiles),("LLM-generated",llm_smiles),("Diffusion Model + LLM",diff_smiles)]
scorer=SAScorer()
with open(OUT,"w",newline="") as f:
    w=csv.writer(f); w.writerow(["group","metric","value"])
    for gname,gen in SRC:
        n=ok=0
        for smi in gen():
            n+=1
            try:
                w.writerow([gname,"BR-SAScore",float(scorer.calculateScore(smi)[0])]); ok+=1
            except Exception:
                pass
            if n%20000==0: print(f"  {gname}: {n} processed",flush=True)
        print(f"{gname}: {ok}/{n} scored",flush=True)
print("BR-SAScore DONE ->",OUT,flush=True)
