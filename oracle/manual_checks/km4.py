import sys, numpy as np, inspect
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from pc_stub import load, lloyd
S = "/Users/plwms/Documents/dissertacao/.oracle_sandbox/"
def f(id): import glob; return [p for p in glob.glob(S + id + "/*.py") if " 2.py" not in p][0]
# Gemma run 2: run_kmeans(data, k, iterations=5), init data[:k], empty -> zeros
m = load(f("RQ2_Gemma_FS_pycompss_kmeans_run2"))
d = [[1, 2], [1, 1], [5, 8], [6, 9], [1, 0], [10, 10], [11, 12], [12, 11]]
print("Gemma run2  max|spec-Lloyd| =", np.abs(np.array(m.run_kmeans(d, 3), float) - lloyd(d, d[:3], 5, "zero")).max())
# Gemma run 3: run_kmeans(data, k=3, iterations=5), init data[:k], empty -> keep
m = load(f("RQ2_Gemma_FS_pycompss_kmeans_run3"))
d = [[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]]
print("Gemma run3  max|spec-Lloyd| =", np.abs(np.array(m.run_kmeans(d), float) - lloyd(d, d[:3], 5)).max())
# Gemma run 5: one pass from random init; compare with one exact Lloyd step from the same init
m = load(f("RQ2_Gemma_FS_pycompss_kmeans_run5"))
np.random.seed(7); got = m.run_kmeans(1000, 3)
np.random.seed(7); X = np.random.rand(1000, 2); C0 = np.random.rand(3, 2)
print("Gemma run5  max|spec-Lloyd(1 step)| =", np.abs(got - lloyd(X, C0, 1)).max())
