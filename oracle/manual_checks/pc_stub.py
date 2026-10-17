import sys, types, importlib.util, numpy as np
def load(path):
    pc = types.ModuleType("pycompss"); api = types.ModuleType("pycompss.api")
    t = types.ModuleType("pycompss.api.task"); a = types.ModuleType("pycompss.api.api"); par = types.ModuleType("pycompss.api.parameter")
    t.task = lambda *x, **k: (lambda f: f)
    a.compss_wait_on = lambda x: x; a.compss_barrier = lambda *x, **k: None
    for n in ["IN", "OUT", "INOUT", "FILE_IN", "FILE_OUT", "FILE_INOUT", "COLLECTION_IN", "COLLECTION_OUT"]:
        setattr(par, n, None)
    for n, m in [("pycompss", pc), ("pycompss.api", api), ("pycompss.api.task", t), ("pycompss.api.api", a), ("pycompss.api.parameter", par)]:
        sys.modules[n] = m
    for n in ["binary", "mpi", "constraint", "multinode", "on_failure", "decaf", "ompss", "software", "container"]:
        m = types.ModuleType("pycompss.api." + n); setattr(m, n, lambda *x, **k: (lambda f: f)); sys.modules["pycompss.api." + n] = m
    spec = importlib.util.spec_from_file_location("spec", path); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod
def lloyd(X, C, it, empty="keep"):
    X = np.asarray(X, float); C = np.asarray(C, float).copy()
    for _ in range(it):
        lab = ((X[:, None] - C[None]) ** 2).sum(-1).argmin(1)
        C = np.array([X[lab == j].mean(0) if (lab == j).any() else (C[j] if empty == "keep" else np.zeros(X.shape[1])) for j in range(len(C))])
    return C
