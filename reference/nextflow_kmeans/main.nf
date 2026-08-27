#!/usr/bin/env nextflow
nextflow.enable.dsl=2

// Generates one data fragment and saves it as a .npy file.
// Each fragment is an independent process — Nextflow runs all of them in parallel.
process generate_fragment {
    input:
        tuple val(frag_id), val(n_points), val(seed)
    output:
        path "fragment_${frag_id}.npy"
    script:
    """
    python3 << 'PYEOF'
import numpy as np
np.random.seed(${seed})
if "${params.mode}" == "normal":
    mat = np.random.normal(0, 1, (${n_points}, ${params.dimensions}))
else:
    mat = np.random.random((${n_points}, ${params.dimensions}))
mat -= mat.min()
mx = mat.max()
if mx > 0:
    mat /= mx
np.save("fragment_${frag_id}.npy", mat)
PYEOF
    """
}

// Receives all fragments (fan-in via .collect()) and runs the full K-Means algorithm.
// In PyCompSs/Parsl the partial_sum step is also parallelised across fragments;
// in Nextflow that level of parallelism would require a feedback-loop channel
// (nextflow.preview.recursion) because centres must be updated between iterations.
// Here the iterative loop runs inside Python while fragment generation stays parallel.
process run_kmeans {
    publishDir params.outdir, mode: 'copy'

    input:
        path fragment_files
        val  num_centres
        val  iterations
        val  seed
        val  epsilon
    output:
        path "centres.npy"
        path "results.txt"
    script:
    """
    python3 << 'PYEOF'
import numpy as np, glob, time
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import paired_distances

data = [np.load(f) for f in sorted(glob.glob("fragment_*.npy"))]
dims = data[0].shape[1]
num_centres = ${num_centres}
max_iter    = ${iterations}
seed        = ${seed}
epsilon     = ${epsilon}

np.random.seed(seed)
centres     = np.array([np.random.random(dims) for _ in range(num_centres)])
old_centres = None
iteration   = 0
start       = time.time()

while True:
    if old_centres is not None:
        dist = paired_distances(centres, old_centres).sum()
        if dist < epsilon**2 or iteration >= max_iter:
            break
    print("Iteration", iteration + 1, "/", max_iter)
    old_centres = centres.copy()
    sums   = np.zeros((num_centres, dims))
    counts = np.zeros(num_centres, dtype=int)
    for frag in data:
        closest = pairwise_distances(frag, centres).argmin(axis=1)
        for c in range(num_centres):
            mask = closest == c
            sums[c]   += frag[mask].sum(axis=0)
            counts[c] += mask.sum()
    for c in range(num_centres):
        if counts[c] > 0:
            centres[c] = sums[c] / counts[c]
    iteration += 1

elapsed = time.time() - start
np.save("centres.npy", centres)

with open("results.txt", "w") as out:
    print("Iterations completed:", iteration, file=out)
    print("Total time (s):      ", round(elapsed, 4), file=out)
    print("Centres:", file=out)
    print(centres, file=out)

print("CENTRES:")
print(centres)
print("Total time (s):", round(elapsed, 4))
PYEOF
    """
}

workflow {
    pts = params.numpoints.intdiv(params.fragments)

    frags_ch = Channel
        .of(0..<params.fragments)
        .map { i -> tuple(i, pts, params.seed + i) }

    fragments = generate_fragment(frags_ch)

    run_kmeans(
        fragments.collect(),
        params.num_centres,
        params.iterations,
        params.seed,
        params.epsilon
    )
}
