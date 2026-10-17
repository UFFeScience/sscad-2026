#!/usr/bin/env python3
"""Map every successful row (Funcionou?=1) of analises_workflows_completo.csv
to the exact file that executed successfully:

  - raw rows      -> the generated candidate itself
  - rerun, run 2-5 -> the repaired copy in .rerun_simple_run2to5_sandbox/
  - rerun, run 1   -> the repaired copy in .rq2_runs/rerun-simple*/<slug>/

Writes oracle/manifest.csv. Rows whose executed file cannot be located are
kept with exec_path empty so they show up in the report instead of vanishing.
"""
import csv
import re
from pathlib import Path

DISS = Path(__file__).resolve().parents[4]          # .../dissertacao
SSCAD = DISS / "sscad"
COMPLETO = SSCAD / "analises" / "analises_workflows_completo.csv"
OUT = Path(__file__).resolve().parent / "manifest.csv"

MODEL_KEY = {"GPT 5.5": "GPT", "Deepseek-V4": "Deepseek",
             "Command-A-Plus-05-2026": "Command", "Gemma-4-31B-IT": "Gemma"}
FW = {"Parsl": "parsl", "Nextflow": "nextflow", "PyCOMPSs": "pycompss"}
FW_SHORT = {"parsl": "parsl", "nextflow": "nf", "pycompss": "pycompss"}


def tag_of(candidate, algo, run):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", f"{candidate}-{algo}-{run}")


def find_run2to5_fix(cand, algo_disp, run):
    root = DISS / ".rerun_simple_run2to5_sandbox"
    for a in (algo_disp, algo_disp.lower()):
        p = root / tag_of(cand, a, run) / Path(cand).name
        if p.exists():
            return p
    return None


def find_run1_fix(r, model, fw, algo):
    slug = f"{model.lower()}-{r['Prompt'].lower()}-{algo}-{FW_SHORT[fw]}"
    if r["Tema"] == "RQ7":
        dirs = [DISS / ".rq2_runs/rerun-simple-rq7" / (slug + "-rq7"),
                DISS / ".rq2_runs/rerun-simple-rq7" / slug,
                DISS / ".rq2_runs/rerun-simple" / (slug + "-rq7")]
        if fw == "nextflow":
            dirs.append(DISS / ".rq2_runs/rerun-simple-rq7" / slug.replace("-nf", "-nextflow"))
    else:
        dirs = [DISS / ".rq2_runs/rerun-simple" / slug]
    name = Path(r["candidate_path"]).name
    for d in dirs:
        if not d.is_dir():
            continue
        p = d / name
        if p.exists():
            return p
        code = [f for f in d.iterdir() if f.suffix in (".py", ".nf")]
        if len(code) == 1:
            return code[0]
    return None


def main():
    rows = [r for r in csv.DictReader(open(COMPLETO)) if r["Funcionou?"] == "1"]
    out = []
    for r in rows:
        model = MODEL_KEY[r["LLM"]]
        fw = FW[r["Linguagem"]]
        algo = r["Algoritmo"].lower()
        cand = r["candidate_path"]
        if r["Fonte"] == "raw":
            p = SSCAD / cand
            exec_path = p if p.exists() else None
        elif r["Run"] == "1":
            exec_path = find_run1_fix(r, model, fw, algo)
        else:
            exec_path = find_run2to5_fix(cand, r["Algoritmo"], r["Run"])
        rid = f"{r['Tema']}_{model}_{r['Prompt']}_{fw}_{algo}_run{r['Run']}"
        out.append({
            "id": rid, "tema": r["Tema"], "model": model, "prompt": r["Prompt"],
            "framework": fw, "algorithm": algo, "run": r["Run"], "fonte": r["Fonte"],
            "candidate_path": cand,
            "exec_path": str(exec_path.relative_to(DISS)) if exec_path else "",
        })
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    missing = [o for o in out if not o["exec_path"]]
    print(f"{len(out)} rows, {len(missing)} without a located exec file")
    for m in missing:
        print("  MISSING", m["id"], m["candidate_path"])


if __name__ == "__main__":
    main()
