#!/usr/bin/env python3
"""Re-execute every successful specification (oracle/manifest.csv) with the
result-capture hook enabled, so the correctness oracles (oracle.py) can check
what the workflow actually computed.

Invocation is NOT reinvented here: it reuses the exact logic of the original
verification harness (verify_run2to5.py / rerun_simple_lib.py) -- sidecar .sh
parsing, CLI-convention probing for RQ2, fixed arguments for RQ7 -- on the
same file that passed. The only additions are:
  * a `.oracle_capture` marker in the sandbox + PYTHONPATH pointing at the
    capture hook (host processes: Parsl, HTEX workers, Nextflow tasks);
  * for PyCOMPSs, the hook is copied into the container's dist-packages as a
    .pth so the COMPSs master and workers load it too;
  * companion files (data files the script names, bin/, nextflow.config) are
    copied from the directories the script originally ran in.

Results: one JSON line per row in oracle/exec_results.jsonl (resumable).
"""
import argparse
import csv
import json
import os
import re
import shlex
import shutil
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
AUTOMATION = HERE.parent
sys.path.insert(0, str(AUTOMATION))
import rerun_simple_lib as rlib  # noqa: E402  (also serializes docker runs with a flock)
import verify_run2to5 as base  # noqa: E402

DISS = HERE.parents[3]
SSCAD = DISS / "sscad"
CAPTURE = HERE / "capture"
SANDBOX_ROOT = DISS / ".oracle_sandbox"
RESULTS = HERE / "exec_results.jsonl"

# verify_run2to5 derives these from its own location, which went stale when the
# study moved under sscad/; point them back at the real venv and shims.
base.SHIM_DIR = DISS / ".rq2_runs" / "rerun-simple" / "_shim"
base.VENV_PYTHON = DISS / ".venv" / "bin" / "python"
base.NATIVE_EXTRA_PATH = f"{base.SHIM_DIR}:{base.VENV_PYTHON.parent}"

_orig_run_cmd = base.run_cmd


def run_cmd_files(cmd, cwd, timeout, extra_path=None, shell=False, use_session=True):
    """base.run_cmd, but stdout/stderr go to files and only the main process is
    awaited: leftover HTEX interchange/worker processes (slower to exit under
    the profiler) otherwise keep the pipes open until the timeout fires."""
    import subprocess
    import tempfile
    if not use_session:          # docker CLI: keep the original behaviour
        return _orig_run_cmd(cmd, cwd, timeout, extra_path, shell, use_session)
    env = os.environ.copy()
    if extra_path:
        env["PATH"] = f"{extra_path}:{env.get('PATH','')}"
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("GMX_MAXBACKUP", "-1")
    t0 = time.time()
    with tempfile.TemporaryFile("w+") as fo, tempfile.TemporaryFile("w+") as fe:
        proc = subprocess.Popen(cmd, cwd=str(cwd), env=env, stdout=fo, stderr=fe, text=True,
                                shell=shell, start_new_session=True)
        timed_out = False
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            base._kill_process_group(proc)
            base._kill_by_cwd(cwd)
        fo.seek(0)
        fe.seek(0)
        out, err = fo.read(), fe.read()
    code = -1 if timed_out else proc.returncode
    return code, out, err, time.time() - t0, timed_out


base.run_cmd = run_cmd_files

# Profiling adds overhead; give every row more headroom than the original pass.
base.KMEANS_TO = 300
base.MATMUL_TO = 300
base.LYSOZYME_TO = 600
base.NEXTFLOW_EXTRA_TO = 60

os.environ["PYTHONPATH"] = str(CAPTURE) + (":" + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")


def run_pycompss_with_capture(sandbox, script_rel, args_tokens, timeout):
    """verify_run2to5.run_pycompss_file + capture hook installed in the image."""
    if not base.ensure_docker_healthy():
        return -4, "", "docker daemon unavailable", 0.0, False, ["docker", "run", "..."]
    safe_tokens = [a.split()[0] if " " in a else a for a in args_tokens]
    inner_args = " ".join(shlex.quote(a) for a in safe_tokens)
    inner = (
        "cp /oracle_capture/oracle_capture_hook.py /oracle_capture/oracle_capture.pth /usr/local/lib/python3.8/dist-packages/; "
        "echo /work > /oracle_root; mkdir -p /workspace/kmeans /workspace/matmul /workspace/lysozyme; "
        "service ssh start >/dev/null 2>&1; "
        "ssh -o StrictHostKeyChecking=no localhost true >/dev/null 2>&1; "
        "command -v gmx_mpi >/dev/null || { printf '#!/bin/bash\\nexec gmx \"$@\"\\n' > /usr/local/bin/gmx_mpi && chmod +x /usr/local/bin/gmx_mpi; }; "
        "command -v mpirun >/dev/null || { printf '#!/bin/bash\\nif [ \"$1\" = \"-np\" ]; then shift 2; fi\\nexec \"$@\"\\n' > /usr/local/bin/mpirun && chmod +x /usr/local/bin/mpirun; }; "
        f"runcompss --python_interpreter=python3 {shlex.quote(script_rel)} {inner_args}; "
        "rc=$?; "
        "for j in /root/.COMPSs/*/jobs/job*_NEW.err /root/.COMPSs/*/jobs/job*.err; do "
        "  if [ -s \"$j\" ]; then echo \"--- $j ---\"; cat \"$j\"; fi; "
        "done 2>/dev/null; "
        "exit $rc"
    )
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", f"oracle_{sandbox.name}")[:120]
    cmd = ["docker", "run", "--rm", "--name", name,
           "-v", f"{sandbox}:/work", "-v", f"{CAPTURE}:/oracle_capture:ro",
           "-w", "/work", base.DOCKER_IMAGE, "bash", "-lc", inner]
    code, out, err, dur, timed_out = base.run_cmd(cmd, sandbox, timeout, use_session=False)
    if timed_out:
        for k in (["docker", "kill", name], ["docker", "rm", "-f", name]):
            try:
                import subprocess
                subprocess.run(k, capture_output=True, timeout=20)
            except Exception:
                pass
    time.sleep(8)
    return code, out, err, dur, timed_out, cmd


rlib._orig_run_pycompss_file = run_pycompss_with_capture  # keeps rlib's cross-process flock


REF_NAME_RE = re.compile(r"[\w.\-/]+\.(?:csv|txt|npy|npz|dat|json|pdb|mdp|top|gro|itp|sh|py|in)\b")


def copy_companions(src_dirs, script_text, sandbox, exec_name):
    """Copy files the script names (by basename) from the directories it ran in,
    plus bin/ and nextflow.config -- inputs only, never another workflow."""
    names = {Path(m).name for m in REF_NAME_RE.findall(script_text)}
    # local modules the script imports (e.g. a parsl_config.py next to it)
    names |= {m + ".py" for m in re.findall(r"^\s*(?:from|import)\s+([A-Za-z_]\w*)", script_text, re.M)}
    for d in src_dirs:
        if not d or not d.is_dir():
            continue
        for extra in ("bin", "templates"):
            if (d / extra).is_dir() and not (sandbox / extra).exists():
                shutil.copytree(d / extra, sandbox / extra)
        for f in ("nextflow.config",):
            if (d / f).exists() and not (sandbox / f).exists():
                shutil.copy(d / f, sandbox / f)
        for n in names:
            p = d / n
            if n != exec_name and p.is_file() and not (sandbox / n).exists():
                shutil.copy(p, sandbox / n)


def run_row(row):
    rid = row["id"]
    sandbox = SANDBOX_ROOT / rid
    if sandbox.exists():
        shutil.rmtree(sandbox)
    sandbox.mkdir(parents=True)
    (sandbox / ".oracle_capture").touch()

    cand = SSCAD / row["candidate_path"]
    exec_path = DISS / row["exec_path"]
    real_name = cand.name
    shutil.copy(exec_path, sandbox / real_name)
    base.seed_fixtures(sandbox)
    text = exec_path.read_text(errors="ignore")
    src_dirs = [exec_path.parent, cand.parent]
    if row["run"] == "1":
        src_dirs.append(DISS / ".rq2_runs" / row["model"])
    copy_companions(src_dirs, text, sandbox, real_name)

    fw, algo = row["framework"], row["algorithm"]
    is_rq7 = row["tema"] == "RQ7"
    is_ablation = "ablation" in Path(row["candidate_path"]).parts
    t0 = time.time()
    try:
        if not is_rq7:
            probe_to = base.KMEANS_TO + (60 if fw == "pycompss" else 0)
            tokens = None if is_ablation else base.parse_sidecar_sh(cand.parent, f"{fw}-{algo}", fw)
            if tokens is None:
                code, out, err, dur, to, cmd, _ = base.run_fallback_invocations(fw, sandbox, real_name, algo, probe_to)
            else:
                tokens = base.substitute_script_token(tokens, real_name, fw)
                if fw == "nextflow":
                    i = tokens.index("run") + 1 if "run" in tokens else 1
                    t = (base.LYSOZYME_TO if algo == "lysozyme" else base.KMEANS_TO) + base.NEXTFLOW_EXTRA_TO
                    code, out, err, dur, to, cmd = base.run_nextflow_file(sandbox, real_name, tokens[i + 1:], t)
                elif fw == "parsl":
                    t = base.LYSOZYME_TO if algo == "lysozyme" else base.KMEANS_TO
                    code, out, err, dur, to, cmd = base.run_parsl_file(sandbox, real_name, tokens[2:], t)
                else:
                    t = (base.LYSOZYME_TO if algo == "lysozyme" else base.KMEANS_TO) + 60
                    code, out, err, dur, to, cmd = base.run_pycompss_file(sandbox, real_name, tokens[2:], t)
        else:
            t = base.LYSOZYME_TO if algo == "lysozyme" else base.KMEANS_TO
            if fw == "nextflow":
                t += base.NEXTFLOW_EXTRA_TO
                args = (["--dataset", str(sandbox / "dataset"), "--config", str(sandbox / "config"), "--outdir", "results"]
                        if algo == "lysozyme" else [])
                code, out, err, dur, to, cmd = base.run_nextflow_file(sandbox, real_name, args, t)
                if code != 0 and not to and algo != "lysozyme":
                    g = base.guess_nf_params(text)
                    if g:
                        ra = [x for k, v in g.items() for x in (f"--{k}", v)]
                        c2 = base.run_nextflow_file(sandbox, real_name, ra, t)
                        if c2[0] == 0:
                            code, out, err, dur, to, cmd = c2
            else:
                args = ([str(sandbox / "config"), str(sandbox / "dataset"), str(sandbox / "output")]
                        if algo == "lysozyme" else [])
                code, out, err, dur, to, cmd = base.run_parsl_file(sandbox, real_name, args, t)
    except Exception as e:
        code, out, err, dur, to, cmd = -3, "", f"harness error: {e}\n{traceback.format_exc()}", time.time() - t0, False, []

    status = base.classify_artifacts(sandbox, algo, code, out, err) if not to else "timeout"
    # keep the full logs on disk for the lysozyme oracle and for manual review
    (sandbox / ".oracle_stdout.log").write_text(out)
    (sandbox / ".oracle_stderr.log").write_text(err)
    return {**row, "exec_status": status, "exit_code": code, "timed_out": to,
            "duration_s": round(dur, 1), "cmd": cmd if isinstance(cmd, list) else [cmd],
            "sandbox": str(sandbox)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--filter", default="")
    ap.add_argument("--redo", action="store_true", help="ignore rows already in exec_results.jsonl")
    a = ap.parse_args()
    rows = list(csv.DictReader(open(HERE / "manifest.csv")))
    if a.filter:
        rows = [r for r in rows if re.search(a.filter, r["id"])]
    done = set()
    if RESULTS.exists() and not a.redo:
        done = {json.loads(l)["id"] for l in open(RESULTS)}
    pending = [r for r in rows if r["id"] not in done]
    print(f"{len(rows)} selected, {len(pending)} to run", flush=True)
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(run_row, r): r for r in pending}
        for i, f in enumerate(as_completed(futs), 1):
            r = futs[f]
            try:
                rec = f.result()
            except Exception as e:
                rec = {**r, "exec_status": "harness-crash", "note": str(e)}
            with open(RESULTS, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(f"[{i}/{len(pending)}] {rec['id']} -> {rec['exec_status']} ({rec.get('duration_s')}s)", flush=True)


if __name__ == "__main__":
    main()
