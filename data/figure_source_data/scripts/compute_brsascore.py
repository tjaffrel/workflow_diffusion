"""Compute BR-SAScore for the 3 linker sources (same inputs as reviewer_br_sascore.py)
and write a long-format CSV (group, metric, value) with metric 'BR-SAScore'."""
import csv, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
from BRSAScore import SAScorer

DIFF = Path("/home/theoj/project/diffusion/synth_scoring")
OUT  = Path("/home/theoj/project/articles/mofgen_natcom_2026/SourceData/Figure2/Figure2bc_BRSAScore.csv")

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
