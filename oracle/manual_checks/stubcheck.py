"""Run a Parsl MatMul spec with Parsl stubbed out (apps called directly) and
check the C it computes against A.B, assembled from its own block grids."""
import sys, types, importlib.util, numpy as np
class Done:
    def __init__(s, v): s.v = v
    def result(s): return s.v
    def done(s): return True
def unwrap(x): return x.v if isinstance(x, Done) else x
def app(f=None, **kw):
    def deco(fn):
        def w(*a, **k):
            return Done(fn(*[unwrap(x) for x in a], **{kk: unwrap(v) for kk, v in k.items()}))
        return w
    return deco(f) if f else deco
parsl = types.ModuleType("parsl"); parsl.load = lambda *a, **k: None; parsl.clear = lambda: None
parsl.python_app = app
for name in ["parsl.app", "parsl.app.app", "parsl.configs", "parsl.configs.local_threads", "parsl.config", "parsl.executors"]:
    m = types.ModuleType(name); sys.modules[name] = m
sys.modules["parsl"] = parsl
sys.modules["parsl.app.app"].python_app = app; sys.modules["parsl.app.app"].bash_app = app
sys.modules["parsl.configs.local_threads"].config = None
sys.modules["parsl.config"].Config = lambda *a, **k: None
sys.modules["parsl.executors"].ThreadPoolExecutor = lambda *a, **k: None
sys.modules["parsl.executors"].HighThroughputExecutor = lambda *a, **k: None
spec = importlib.util.spec_from_file_location("spec", sys.argv[1]); mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
# re-run main's logic, keeping its grids
captured = {}
orig_dot = mod.dot
def dot_spy(A, B, C, *rest):
    out = orig_dot(A, B, C, *rest)
    captured.update(A=A, B=B, C=C)
    return out
mod.dot = dot_spy
n, e = int(sys.argv[2]), int(sys.argv[3])
import inspect
if inspect.signature(mod.main).parameters:
    mod.main(num_blocks=n, elems_per_block=e, seed=0)
else:
    mod.main()
blk = lambda G: np.block([[unwrap(x) for x in row] for row in G])
A, B, C = blk(captured["A"]), blk(captured["B"]), blk(captured["C"])
err = np.linalg.norm(C - A @ B) / (np.linalg.norm(A) * np.linalg.norm(B))
print(f"{sys.argv[1].split('/')[-2]}: A{A.shape} B{B.shape} C{C.shape}  ||C-AB||/(||A||·||B||) = {err:.2e}")
