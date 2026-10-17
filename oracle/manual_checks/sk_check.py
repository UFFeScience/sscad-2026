"""Direct check for specs that call scikit-learn KMeans on an input file:
load the file the task actually read, take the cluster_centers_ the run
produced, and test the Lloyd fixed point on that data."""
import sys, json, glob, numpy as np, pandas as pd
sb, datafile = sys.argv[1], sys.argv[2]
p = glob.glob(f"{sb}/work/**/{datafile}", recursive=True) or glob.glob(f"{sb}/{datafile}")
raw = open(p[0]).read().splitlines()
try:
    X = np.loadtxt(p[0], delimiter=",")
except Exception:
    X = pd.read_csv(p[0]).select_dtypes("number").to_numpy()
X = np.atleast_2d(X).astype(float)
recs = [json.loads(l) for f in glob.glob(f"{sb}/.oracle_out/*.jsonl") for l in open(f)]
cc = [r for r in recs if "cluster_centers_" in r["var"]]
C = np.load(f"{sb}/.oracle_out/{cc[-1]['file']}")
lab = ((X[:, None] - C[None]) ** 2).sum(-1).argmin(1)
U = np.array([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(len(C))])
print(f"{sb.split('/')[-1]}: n={len(X)} k={len(C)} max|U(C)-C|={np.abs(U-C).max():.2e}")
