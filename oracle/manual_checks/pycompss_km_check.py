import sys, types, importlib.util, numpy as np
pc = types.ModuleType("pycompss"); api = types.ModuleType("pycompss.api")
task_m = types.ModuleType("pycompss.api.task"); api_m = types.ModuleType("pycompss.api.api"); par = types.ModuleType("pycompss.api.parameter")
task_m.task = lambda *a, **k: (lambda f: f)
api_m.compss_wait_on = lambda x: x; api_m.compss_barrier = lambda *a, **k: None
par.IN = par.OUT = par.INOUT = par.FILE_IN = par.FILE_OUT = par.COLLECTION_IN = None
for n, m in [("pycompss", pc), ("pycompss.api", api), ("pycompss.api.task", task_m), ("pycompss.api.api", api_m), ("pycompss.api.parameter", par)]:
    sys.modules[n] = m
spec = importlib.util.spec_from_file_location("spec", sys.argv[1]); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
rng = np.random.default_rng(0); data = rng.random((10000, 10)); k, it = 3, 5
got = np.array(mod.kmeans_workflow(data, k=k, iterations=it))
C = data[:k].copy()                      # same init as the spec
for _ in range(it):
    lab = ((data[:, None, :] - C[None]) ** 2).sum(-1).argmin(1)
    C = np.array([data[lab == j].mean(0) if (lab == j).any() else np.zeros(data.shape[1]) for j in range(k)])
print("max |spec - Lloyd| =", np.abs(got - C).max())
