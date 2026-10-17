"""Result-capture hook for the correctness oracles.

Loaded at interpreter start-up in every Python process a workflow spawns
(the Parsl driver and its HTEX workers, every Nextflow task that runs
python, and the PyCOMPSs master/workers inside the container). It never
changes what the workflow computes: it only observes, through a profiler,
the 2-D numeric arrays held by the *workflow's own* functions when they
return (arguments, locals and return value), and writes a copy of each
distinct one to disk for the post-hoc oracle (oracle.py).

Activation: a process captures only if its cwd is inside a directory that
contains a `.oracle_capture` marker file (the per-row sandbox). Arrays are
written to <sandbox>/.oracle_out/<pid>_<seq>.npy with one JSON line of
metadata per array in <sandbox>/.oracle_out/<pid>.jsonl.

"User code" = any code object whose file lives under the sandbox (or is an
inline '<stdin>' / '<string>' script, as Nextflow heredocs are). Library code
(numpy, sklearn, parsl, pycompss, stdlib) is never inspected.
"""
import os
import sys
import threading
import time

_MAX_ELEMS = 4096 * 4096          # skip anything bigger than 16.7M elements
_MAX_BYTES_PER_PROC = 3 * 1024 ** 3
_MAX_SNAPSHOTS_PER_CODE = 4000    # a hot helper called millions of times stops being inspected
_MIN_ELEMS = 4


def _find_root():
    # Inside the PyCOMPSs container, workers run in COMPSs' own scratch dirs,
    # so the sandbox is announced through /oracle_root instead of the cwd.
    try:
        with open("/oracle_root") as f:
            r = f.read().strip()
        if r and os.path.exists(os.path.join(r, ".oracle_capture")):
            return r
    except Exception:
        pass
    d = os.path.abspath(os.getcwd())
    while True:
        if os.path.exists(os.path.join(d, ".oracle_capture")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            return None
        d = parent


def install():
    root = _find_root()
    if root is None:
        return
    try:
        import numpy as np  # noqa: F401
    except Exception:
        return
    _Capture(root).start()


class _Capture:
    def __init__(self, root):
        import numpy as np
        self.np = np
        self.root = root
        self.out = os.path.join(root, ".oracle_out")
        os.makedirs(self.out, exist_ok=True)
        self.pid = os.getpid()
        # pids get reused by short-lived workers (COMPSs, Nextflow tasks):
        # make every process's files unique with a random token
        self.tag = "%d_%s" % (self.pid, os.urandom(4).hex())
        self.meta_path = os.path.join(self.out, "%s.jsonl" % self.tag)
        self.lock = threading.RLock()
        self.seen = {}
        self.callno = 0
        self.tls = threading.local()
        self.dupcount = {}
        self.seq = 0
        self.bytes = 0
        self.per_code = {}
        self.user_cache = {}
        self.disabled = False

    def start(self):
        if hasattr(os, "register_at_fork"):
            os.register_at_fork(after_in_child=self._after_fork)
        sys.setprofile(self._prof)
        threading.setprofile(self._prof)

    # -- filtering ---------------------------------------------------------
    def _is_user(self, code):
        r = self.user_cache.get(code)
        if r is not None:
            return r
        fn = code.co_filename
        if fn in ("<stdin>", "<string>"):
            r = True
        elif fn.startswith("<"):
            r = False
        else:
            a = os.path.abspath(fn)
            r = a.startswith(self.root + os.sep) and "site-packages" not in a \
                and "dist-packages" not in a and "oracle_capture_hook" not in a
        self.user_cache[code] = r
        return r

    def _prof(self, frame, event, arg):
        if event != "return" or self.disabled:
            return
        code = frame.f_code
        if not self._is_user(code):
            return
        n = self.per_code.get(code, 0)
        if n >= _MAX_SNAPSHOTS_PER_CODE:
            return
        self.per_code[code] = n + 1
        try:
            fname = code.co_name
            loc = list(frame.f_locals.items())
            # scalar context (block indices, seeds, sizes) recorded with every
            # array of this frame, so the oracle can place blocks in a grid
            ctx = {}
            for k, v in loc:
                if isinstance(v, bool) or v is None:
                    continue
                if isinstance(v, (int, float)) or (isinstance(v, str) and len(v) <= 64):
                    ctx[k] = v
                    if len(ctx) >= 24:
                        break
            with self.lock:
                self.callno += 1
                call = self.callno
            self.tls.ctx = ctx
            self.tls.call = call
            for k, v in loc:
                self._visit(v, fname, k, 0)
            self._visit(arg, fname, "<return>", 0)
        except Exception:
            pass

    # -- value walking -----------------------------------------------------
    def _resolve(self, v):
        """Parsl AppFuture / concurrent Future that already finished -> value."""
        if hasattr(v, "done") and hasattr(v, "result") and callable(getattr(v, "done", None)):
            try:
                if v.done() and v.exception() is None:
                    return v.result()
            except Exception:
                return None
            return None
        return v

    def _visit(self, v, fname, var, depth):
        np = self.np
        if depth > 3 or v is None:
            return
        v = self._resolve(v)
        if v is None:
            return
        if isinstance(v, np.ndarray):
            if v.ndim == 2:
                self._save(v, fname, var, "array")
            return
        if hasattr(v, "cluster_centers_"):
            try:
                self._save(np.asarray(v.cluster_centers_), fname, var + ".cluster_centers_", "array")
            except Exception:
                pass
            return
        if type(v).__name__ == "DataFrame":
            try:
                a = v.select_dtypes("number").to_numpy()
                self._save(a, fname, var, "dataframe")
            except Exception:
                pass
            return
        if isinstance(v, dict):
            for k2, v2 in list(v.items())[:256]:
                self._visit(v2, fname, "%s[%r]" % (var, k2), depth + 1)
            return
        if isinstance(v, (list, tuple)) and v:
            if len(v) > 4096:
                # long lists: only numeric rows (pure-Python datasets) are recorded
                first = v[0]
                if len(v) > 200000 or not (isinstance(first, (list, tuple)) and first
                                           and isinstance(first[0], (int, float))):
                    return
            items = [self._resolve(x) for x in v]
            # list of lists of numbers -> matrix
            if all(isinstance(x, (list, tuple)) for x in items) and len(items) >= 2:
                try:
                    if all(isinstance(e, (int, float)) for x in items for e in x):
                        a = np.asarray(items, dtype=float)
                        if a.ndim == 2:
                            self._save(a, fname, var, "pylist")
                            return
                except Exception:
                    pass
                # grid of 2-D blocks -> assembled matrix
                try:
                    if all(isinstance(e, np.ndarray) and e.ndim == 2 for x in items for e in x):
                        self._save(np.block([list(x) for x in items]), fname, var, "grid")
                except Exception:
                    pass
            # list of equal-length 1-D arrays (e.g. centroids kept as a list) -> matrix
            vecs = [x for x in items if isinstance(x, np.ndarray) and x.ndim == 1]
            if len(vecs) >= 2 and len(vecs) == len(items) and len({v.shape[0] for v in vecs}) == 1:
                try:
                    self._save(np.vstack(vecs), fname, var, "stack1d")
                except Exception:
                    pass
            # list of same-width 2-D arrays (data fragments) -> stacked
            arrs = [x for x in items if isinstance(x, np.ndarray) and x.ndim == 2]
            if len(arrs) >= 2 and len(arrs) == len(items) and len({a.shape[1] for a in arrs}) == 1:
                try:
                    self._save(np.vstack(arrs), fname, var, "vstack")
                except Exception:
                    pass
            for i, x in enumerate(items[:512]):
                self._visit(x, fname, "%s[%d]" % (var, i), depth + 1)

    def _after_fork(self):
        """Forked workers (COMPSs executors, Parsl HTEX pool) inherit this
        object: give the child its own file tag, counter, dedupe table and a
        fresh lock -- the inherited one may have been held by another thread
        at fork time and would deadlock the child."""
        self.lock = threading.RLock()
        self.tls = threading.local()
        self.pid = os.getpid()
        self.tag = "%d_%s" % (self.pid, os.urandom(4).hex())
        self.meta_path = os.path.join(self.out, "%s.jsonl" % self.tag)
        self.seen, self.dupcount, self.seq, self.bytes = {}, {}, 0, 0

    def _save(self, a, fname, var, kind):
        np = self.np
        if a.dtype.kind not in "fiu" or a.size < _MIN_ELEMS or a.size > _MAX_ELEMS:
            return
        if a.ndim != 2 or min(a.shape) < 1:
            return
        a = np.ascontiguousarray(a)
        import hashlib
        h = hashlib.blake2b(a.tobytes(), digest_size=16).hexdigest() + str(a.shape)
        import json
        with self.lock:
            if h in self.seen:
                n = self.dupcount.get(h, 0)
                if n >= 64:
                    return
                self.dupcount[h] = n + 1
                # same array seen again (e.g. passed to another task): log the
                # new role so the oracle can follow the dataflow, no new copy
                rec = {"file": self.seen[h], "pid": self.tag, "seq": None, "t": time.time(),
                       "func": fname, "var": var, "kind": kind, "shape": list(a.shape),
                       "dtype": str(a.dtype), "hash": h, "ctx": getattr(self.tls, "ctx", {}), "call": getattr(self.tls, "call", 0), "dup": True}
                with open(self.meta_path, "a") as f:
                    f.write(json.dumps(rec, default=str) + "\n")
                return
            if self.bytes + a.nbytes > _MAX_BYTES_PER_PROC:
                self.disabled = True
                return
            self.seq += 1
            seq = self.seq
            self.seen[h] = "%s_%d.npy" % (self.tag, seq)
            self.bytes += a.nbytes
        path = os.path.join(self.out, "%s_%d.npy" % (self.tag, seq))
        np.save(path, a)
        rec = {"file": os.path.basename(path), "pid": self.tag, "seq": seq, "t": time.time(),
               "func": fname, "var": var, "kind": kind, "shape": list(a.shape),
               "dtype": str(a.dtype), "hash": h, "ctx": getattr(self.tls, "ctx", {}), "call": getattr(self.tls, "call", 0), "dup": False}
        with self.lock:
            with open(self.meta_path, "a") as f:
                f.write(json.dumps(rec, default=str) + "\n")
