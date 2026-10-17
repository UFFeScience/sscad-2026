#!/usr/bin/env python3
"""Final oracle verdicts for the K-Means/MatMul successes (automatic oracle +
manual direct checks) and their effect on ER / ER+ per table cell.
Writes oracle/final_verdicts.csv and oracle/table_impact.csv."""
import csv, json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SSCAD = HERE.parents[2]

# Manual review of every row the automatic oracle did not PASS (and of rows it
# could not settle). Each entry: id -> (verdict, reason).
MANUAL = {
 # --- placeholders / simulations: no K-Means or MatMul is computed -----------
 **{i: ("FAIL", "placeholder: task only echoes text, nothing is computed") for i in [
    "RQ2_Gemma_FS_nextflow_kmeans_run2", "RQ2_Gemma_FS_nextflow_matmul_run2",
    "RQ2_Gemma_ZS_nextflow_kmeans_run2", "RQ2_Gemma_ZS_nextflow_kmeans_run3",
    "RQ2_Gemma_ZS_nextflow_kmeans_run4", "RQ2_Gemma_ZS_nextflow_kmeans_run5",
    "RQ2_Gemma_ZS_nextflow_matmul_run2", "RQ2_Gemma_ZS_nextflow_matmul_run3",
    "RQ2_Gemma_ZS_nextflow_matmul_run4", "RQ2_Gemma_ZS_nextflow_matmul_run5"]},
 "RQ2_Gemma_FS_parsl_kmeans_run2": ("FAIL", "simulation: sum of random numbers reported as a centroid"),
 "RQ2_Gemma_FS_parsl_matmul_run2": ("FAIL", "simulation: sum of random products reported as a checksum"),
 "RQ2_Gemma_FS_parsl_kmeans_run5": ("FAIL", "simulation: returns random centroids, no data is read"),
 "RQ7_Gemma_FS_nextflow_matmul_run4": ("FAIL", "no process is executed (completed=0)"),
 # --- algorithmic defects ----------------------------------------------------
 "RQ2_Deepseek_FS_pycompss_kmeans_run3": ("FAIL", "chunk-local indices applied to the full dataset"),
 "RQ2_Command_FS_parsl_kmeans_run4": ("FAIL", "integer centroid array truncates every mean"),
 "RQ2_Gemma_FS_pycompss_kmeans_run5": ("FAIL", "unweighted average of per-chunk means (6e-4 off a Lloyd step)"),
 "RQ7_Command_FS_parsl_kmeans_run5": ("FAIL", "loop tests a Parsl Future (always true): no iteration runs"),
 "RQ7_GPT_FS_parsl_matmul_run1": ("FAIL", "partial products never accumulated (||C-AB|| = 0.78 ||A|| ||B||)"),
 "RQ7_Command_FS_parsl_matmul_run5": ("FAIL", "multiply-add result discarded (PyCOMPSs INOUT semantics lost in Parsl)"),
 "RQ2_Gemma_ZS_pycompss_matmul_run1": ("FAIL", "result never synchronised: np.vstack of unresolved futures"),
 "RQ2_Gemma_ZS_pycompss_matmul_run3": ("FAIL", "result never synchronised: np.vstack of unresolved futures"),
 # --- does not execute as generated -----------------------------------------
 "RQ2_Deepseek_FS_nextflow_matmul_run1": ("FAIL", "not executable as generated: declared output A.csv, writes matrix_a.csv"),
 "RQ2_Gemma_FS_nextflow_kmeans_run1": ("FAIL", "not executable as generated: script called by relative path from the work dir"),
 "RQ2_Gemma_FS_nextflow_matmul_run1": ("FAIL", "not executable as generated: script called by relative path from the work dir"),
 # --- automatic oracle could not settle; verified correct by direct check ---
 **{i: ("PASS", "direct check: scikit-learn centres are an exact Lloyd fixed point on the input file") for i in [
    "RQ2_Command_FS_nextflow_kmeans_run2", "RQ2_Command_FS_nextflow_kmeans_run4",
    "RQ2_Deepseek_FS_nextflow_kmeans_run1", "RQ2_Deepseek_FS_nextflow_kmeans_run3",
    "RQ2_GPT_FS_nextflow_kmeans_run1", "RQ2_Gemma_ZS_nextflow_kmeans_run1",
    "RQ2_Deepseek_ZS_nextflow_kmeans_run4"]},
 "RQ2_Deepseek_ZS_parsl_kmeans_run1": ("PASS", "direct check: scikit-learn KMeans on the generated data"),
 "RQ2_Gemma_FS_parsl_kmeans_run3": ("PASS", "direct check: scikit-learn KMeans per task (input partition ignored)"),
 "RQ2_Gemma_FS_pycompss_kmeans_run2": ("PASS", "direct check: identical to Lloyd from the same initialisation"),
 "RQ2_Gemma_FS_pycompss_kmeans_run3": ("PASS", "direct check: identical to Lloyd from the same initialisation"),
 "RQ2_Deepseek_FS_pycompss_kmeans_run5": ("PASS", "direct check: exactly one Lloyd step over all points"),
 "RQ2_GPT_ZS_pycompss_kmeans_run1": ("PASS", "direct check: identical to Lloyd from the same initialisation"),
 "RQ7_Deepseek_FS_parsl_kmeans_run1": ("PASS", "direct check: identical to Lloyd (1e-16)"),
 "RQ7_Deepseek_ZS_parsl_kmeans_run1": ("PASS", "direct check: same parameters and centres as the verified reference"),
 "RQ7_Gemma_ZS_parsl_kmeans_run1": ("PASS", "direct check: same parameters and centres as the verified reference"),
 "RQ2_Deepseek_ZS_pycompss_matmul_run4": ("PASS", "direct check: single np.dot task on rectangular A, B"),
 "RQ2_GPT_ZS_parsl_matmul_run4": ("PASS", "direct check: C.npy equals A.B with A regenerated from its seeds"),
 "RQ7_Deepseek_ZS_parsl_matmul_run4": ("PASS", "direct check: ||C-AB|| = 3e-17 ||A|| ||B||"),
 "RQ7_Gemma_FS_parsl_matmul_run1": ("PASS", "direct check: ||C-AB|| = 5e-17 ||A|| ||B||"),
 "RQ7_Gemma_FS_nextflow_matmul_run3": ("PASS", "direct check: result equals A.B exactly (1x1 grid)"),
}


def is_raw(r):
    """Unrepaired success, as counted in the paper's ER: generated code that ran
    as is -- including run-1 rows re-executed without any code change."""
    note = (r.get("Detalhes", "") + " " + r.get("Observação", "")).lower()
    return r["Fonte"] == "raw" or "sem nenhuma alteração de código" in note


def main():
    auto = {}
    completo = {}
    model_key = {"GPT 5.5": "GPT", "Deepseek-V4": "Deepseek", "Command-A-Plus-05-2026": "Command", "Gemma-4-31B-IT": "Gemma"}
    for r in csv.DictReader(open(SSCAD / "analises" / "analises_workflows_completo.csv")):
        rid = f"{r['Tema']}_{model_key[r['LLM']]}_{r['Prompt']}_{r['Linguagem'].lower()}_{r['Algoritmo'].lower()}_run{r['Run']}"
        completo[rid] = r
    for line in open(HERE / "oracle_progress.jsonl"):
        d = json.loads(line)
        auto[d["id"]] = d["oracle"]
    man = {r["id"]: r for r in csv.DictReader(open(HERE / "manifest.csv"))}
    rows = []
    for rid, r in man.items():
        if r["algorithm"] == "lysozyme":
            continue
        if rid in MANUAL:
            v, why, how = *MANUAL[rid], "manual"
        else:
            v, why, how = auto[rid], "automatic oracle", "automatic"
            assert v == "PASS", (rid, v)  # every non-PASS must have been reviewed
        rows.append({**{k: r[k] for k in ("id", "tema", "model", "prompt", "framework", "algorithm", "run", "fonte")},
                     "verdict": v, "how": how, "reason": why})
    with open(HERE / "final_verdicts.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # ER / ER+ per (task, system, model, prompt): 3 apps x 5 runs = 15 specs
    model_of = {"GPT 5.5": "GPT", "Deepseek-V4": "Deepseek", "Command-A-Plus-05-2026": "Command", "Gemma-4-31B-IT": "Gemma"}
    fw_of = {"Parsl": "parsl", "Nextflow": "nextflow", "PyCOMPSs": "pycompss"}
    raw, plus = defaultdict(int), defaultdict(int)
    for r in csv.DictReader(open(SSCAD / "analises" / "analises_workflows_completo.csv")):
        if r["Funcionou?"] != "1":
            continue
        key = (r["Tema"], fw_of[r["Linguagem"]], model_of[r["LLM"]], r["Prompt"])
        plus[key] += 1
        if is_raw(r):
            raw[key] += 1
    dem_raw, dem_plus = defaultdict(int), defaultdict(int)
    for r in rows:
        if r["verdict"] == "FAIL":
            key = (r["tema"], r["framework"], r["model"], r["prompt"])
            dem_plus[key] += 1
            if is_raw(completo[r["id"]]):
                dem_raw[key] += 1
    out = []
    for key in sorted(set(plus) | set(raw)):
        out.append({"task": key[0], "system": key[1], "model": key[2], "prompt": key[3],
                    "ER_old": round(100 * raw[key] / 15, 1), "ER_new": round(100 * (raw[key] - dem_raw[key]) / 15, 1),
                    "ERplus_old": round(100 * plus[key] / 15, 1), "ERplus_new": round(100 * (plus[key] - dem_plus[key]) / 15, 1),
                    "demoted": dem_plus[key]})
    with open(HERE / "table_impact.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
    n_fail = sum(r["verdict"] == "FAIL" for r in rows)
    print(f"{len(rows)} K-Means/MatMul successes: {len(rows) - n_fail} PASS, {n_fail} FAIL "
          f"({sum(r['how']=='manual' for r in rows)} decided manually)")
    for o in out:
        if o["demoted"]:
            print(f"  {o['task']} {o['system']:9s} {o['model']:8s} {o['prompt']}: ER {o['ER_old']} -> {o['ER_new']}, "
                  f"ER+ {o['ERplus_old']} -> {o['ERplus_new']}")


if __name__ == "__main__":
    main()
