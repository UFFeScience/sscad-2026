#!/usr/bin/env python3
"""Negative controls: deliberately broken copies of the Parsl reference
workflows. Each mutant still executes to completion (so ER would count it as
a success) but computes a wrong result; the oracles must reject all of them.
Results: oracle/mutant_exec_results.jsonl, then `oracle.py --exec` on it.
"""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_oracle_exec as rx  # noqa: E402

base = rx.base
REF = rx.DISS / "sscad-2026" / "reference"
OUT = rx.HERE / "mutant_exec_results.jsonl"

MM = ("parsl_matmul", "matmul.py", ["-b", "3", "-e", "64"])
KM = ("parsl_kmeans", "kmeans.py", ["-n", "1000", "-f", "10"])
MUTANTS = {
    # wrong operand orientation inside every task
    "mm_transposed_B": (MM, "return C + np.dot(A, B)", "return C + np.dot(A, B.T)"),
    # wrong dataflow: pairs A[i][k] with B[j][k] instead of B[k][j]
    "mm_wrong_pairing": (MM, "fused_multiply_add(A[i][k], B[k][j], C[i][j])",
                         "fused_multiply_add(A[i][k], B[j][k], C[i][j])"),
    # lost accumulation: every task overwrites the partial sum
    "mm_no_accumulation": (MM, "return C + np.dot(A, B)", "return np.dot(A, B)"),
    # assignment to the farthest centre
    "km_argmax": (KM, "pairwise_distances(fragment, centres).argmin(axis=1)",
                  "pairwise_distances(fragment, centres).argmax(axis=1)"),
    # wrong mean (off-by-one count)
    "km_bad_mean": (KM, "centres[idx] = sum_[0] / sum_[1]", "centres[idx] = sum_[0] / (sum_[1] + 1)"),
    # centres never move
    "km_no_update": (KM, "centres[idx] = sum_[0] / sum_[1]", "centres[idx] = centres[idx]"),
}


def main():
    OUT.unlink(missing_ok=True)
    for name, ((ref, script, args), old, new) in MUTANTS.items():
        sb = rx.SANDBOX_ROOT / f"MUT_{name}"
        if sb.exists():
            shutil.rmtree(sb)
        shutil.copytree(REF / ref, sb)
        (sb / ".oracle_capture").touch()
        src = (sb / script).read_text()
        assert old in src, (name, old)
        (sb / script).write_text(src.replace(old, new))
        code, out, err, dur, timed, cmd = base.run_parsl_file(sb, script, args, 300)
        status = base.classify_artifacts(sb, ref.split("_")[1], code, out, err) if not timed else "timeout"
        rec = {"id": f"MUT_{name}", "tema": "MUT", "model": "mutant", "prompt": "-", "framework": "parsl",
               "algorithm": ref.split("_")[1], "run": "-", "fonte": "mutant", "exec_status": status,
               "exit_code": code, "timed_out": timed, "duration_s": round(dur, 1), "cmd": cmd, "sandbox": str(sb)}
        with open(OUT, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(rec["id"], status, flush=True)


if __name__ == "__main__":
    main()
