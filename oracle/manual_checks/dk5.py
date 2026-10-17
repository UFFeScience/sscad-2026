import sys, random, numpy as np
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from pc_stub import load, lloyd
m = load("/Users/plwms/Documents/dissertacao/.oracle_sandbox/RQ2_Deepseek_FS_pycompss_kmeans_run5/pycompss-workflow-kmeans.py")
got = {}
m.pprint = lambda x: got.setdefault("C", x)
N, K, D, CH = [int(a) for a in sys.argv[1:5]]
m.main_kmeans(N, K, D, CH)
random.seed(42); pts = [[random.random() for _ in range(D)] for _ in range(N)]
print("all points   :", np.abs(np.array(got["C"]) - lloyd(pts, pts[:K], 1)).max())
used = (N // CH) * CH
print("chunked part :", np.abs(np.array(got["C"]) - lloyd(pts[:used], pts[:K], 1)).max(), f"(uses {used}/{N} points)")
