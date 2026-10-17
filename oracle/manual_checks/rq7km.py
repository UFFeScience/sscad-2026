import sys, types, importlib.util, inspect, numpy as np
from sklearn.metrics.pairwise import paired_distances
class Done:
    def __init__(s, v): s.v = v
    def result(s): return s.v
    def done(s): return True
    def exception(s): return None
def unwrap(x): return x.v if isinstance(x, Done) else x
def app(f=None, **kw):
    def deco(fn):
        def w(*a, **k): return Done(fn(*[unwrap(x) for x in a], **{kk: unwrap(v) for kk, v in k.items()}))
        w.__wrapped__ = fn; return w
    return deco(f) if callable(f) else deco
for name in ["parsl", "parsl.app", "parsl.app.app", "parsl.configs", "parsl.configs.local_threads", "parsl.configs.htex_local",
             "parsl.data_provider", "parsl.data_provider.files", "parsl.dataflow", "parsl.dataflow.futures", "parsl_config"]:
    sys.modules[name] = types.ModuleType(name)
p = sys.modules["parsl"]; p.load = lambda *a, **k: None; p.clear = lambda: None
p.python_app = p.join_app = p.bash_app = app; p.File = object
sys.modules["parsl.app.app"].python_app = app; sys.modules["parsl.app.app"].bash_app = app
sys.modules["parsl.configs.local_threads"].config = None; sys.modules["parsl.configs.htex_local"].config = None
sys.modules["parsl.data_provider.files"].File = object; sys.modules["parsl.dataflow.futures"].AppFuture = Done
sys.modules["parsl_config"].config = None
spec = importlib.util.spec_from_file_location("spec", sys.argv[1]); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
seen = {}
orig = m.kmeans_frag
def spy(fragments, dimensions, num_centres=10, iterations=20, seed=0.0, epsilon=1e-9, arity=50, *a, **k):
    out = orig(fragments, dimensions, num_centres, iterations, seed, epsilon, arity, *a, **k)
    seen.update(F=[np.asarray(unwrap(f)) for f in fragments], d=dimensions, k=num_centres, it=iterations, seed=seed, eps=epsilon, out=unwrap(out))
    return out
m.kmeans_frag = spy
argv = sys.argv[2:]
sys.argv = [sys.argv[1]] + argv
src = open(spec.origin).read()
ns = dict(vars(m)); ns["__name__"] = "__main__"; ns["kmeans_frag"] = spy
exec(compile(src.split("if __name__")[-1].join(["if __name__", ""]) if False else "", "x", "exec"), ns)
if "main" in vars(m):
    sig = inspect.signature(m.main)
    if sig.parameters:
        # same arguments the script's own __main__ block would pass
        import argparse
        args = m.parse_arguments() if hasattr(m, "parse_arguments") else m.parse_args()
        m.main(**vars(args))
    else:
        m.main()
X = np.vstack(seen["F"]); np.random.seed(int(seen["seed"]))
C = np.asarray([np.random.random(seen["d"]) for _ in range(seen["k"])]); old = None; it = 0
while not (old is not None and (np.sum(paired_distances(C, old)) < seen["eps"] ** 2 or it >= seen["it"])):
    old = C.copy(); lab = ((X[:, None] - C[None]) ** 2).sum(-1).argmin(1)
    for j in range(len(C)):
        if (lab == j).any(): C[j] = X[lab == j].mean(0)
    it += 1
print(f"{sys.argv[0].split('/')[-2]}: n={len(X)} k={seen['k']} iters={it}  max|spec-Lloyd|={np.abs(np.asarray(seen['out']) - C).max():.2e}")
