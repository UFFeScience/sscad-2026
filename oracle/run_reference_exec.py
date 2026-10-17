#!/usr/bin/env python3
"""Run the nine expert-developed reference workflows through the same capture
harness, as a sanity check of the oracles (every reference must PASS).
Arguments are the historical ones from analises/performance/invocation_args.csv.
Results: oracle/ref_exec_results.jsonl
"""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_oracle_exec as rx  # noqa: E402  (sets PYTHONPATH, shims, docker capture)

base = rx.base
REF = rx.DISS / "sscad-2026" / "reference"
OUT = rx.HERE / "ref_exec_results.jsonl"

ARGS = {
    ("nextflow", "kmeans"): [], ("nextflow", "matmul"): [],
    ("nextflow", "lysozyme"): ["--dataset", "{sb}/dataset", "--config", "{sb}/config"],
    ("parsl", "kmeans"): ["-n", "1000", "-f", "10"], ("parsl", "matmul"): ["-b", "3", "-e", "64"],
    ("parsl", "lysozyme"): ["{sb}/config", "{sb}/dataset", "{sb}/output"],
    ("pycompss", "kmeans"): ["-n", "1000", "-f", "10", "-i", "10"],
    ("pycompss", "matmul"): ["-b", "4", "-e", "128"],
    ("pycompss", "lysozyme"): ["config", "dataset", "output"],
}


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    if not only:
        OUT.unlink(missing_ok=True)
    for (fw, algo), args in ARGS.items():
        if only and only not in f"{fw}_{algo}":
            continue
        src = REF / f"{fw}_{algo}"
        sb = rx.SANDBOX_ROOT / f"REF_{fw}_{algo}"
        if sb.exists():
            shutil.rmtree(sb)
        shutil.copytree(src, sb)
        (sb / ".oracle_capture").touch()
        (sb / "output").mkdir(exist_ok=True)
        for f in sb.glob("metrics.jsonl"):
            f.unlink()
        script = "main.nf" if fw == "nextflow" else f"{algo}.py"
        a = [x.replace("{sb}", str(sb)) for x in args]
        to = 900
        if fw == "nextflow":
            code, out, err, dur, timed, cmd = base.run_nextflow_file(sb, script, a, to)
        elif fw == "parsl":
            code, out, err, dur, timed, cmd = base.run_parsl_file(sb, script, a, to)
        else:
            code, out, err, dur, timed, cmd = base.run_pycompss_file(sb, script, a, to)
        status = base.classify_artifacts(sb, algo, code, out, err) if not timed else "timeout"
        (sb / ".oracle_stdout.log").write_text(out)
        (sb / ".oracle_stderr.log").write_text(err)
        rec = {"id": f"REF_{fw}_{algo}", "tema": "REF", "model": "reference", "prompt": "-",
               "framework": fw, "algorithm": algo, "run": "-", "fonte": "reference",
               "exec_status": status, "exit_code": code, "timed_out": timed,
               "duration_s": round(dur, 1), "cmd": cmd, "sandbox": str(sb)}
        with open(OUT, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(rec["id"], status, round(dur, 1), flush=True)


if __name__ == "__main__":
    main()
