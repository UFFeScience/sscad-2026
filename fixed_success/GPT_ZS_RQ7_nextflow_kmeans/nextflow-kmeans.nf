nextflow.enable.dsl = 2

params.seed        = 0
params.numpoints   = 100
params.dimensions  = 2
params.num_centres = 5
params.fragments   = 10
params.mode        = 'uniform'
params.iterations  = 20
params.epsilon     = 1e-9
params.arity       = 50
params.outdir      = 'results_kmeans'


process GENERATE_FRAGMENT {

    tag "fragment ${idx}"

    publishDir params.outdir, mode: 'copy'

    input:
    tuple val(idx), val(points), val(dim), val(mode), val(seed)

    output:
    path "fragment_${idx}.npy"

    script:
    """
    python3 - <<'PY'
import numpy as np

idx    = ${idx}
points = ${points}
dim    = ${dim}
mode   = "${mode}"
seed   = ${seed}

rand = {
    "normal":  lambda k: np.random.normal(0, 1, k),
    "uniform": lambda k: np.random.random(k),
}

np.random.seed(seed)
r = rand[mode]

mat = np.asarray([r(dim) for _ in range(points)])

# Normalização entre 0 e 1
mat -= np.min(mat)
mx = np.max(mat)
if mx > 0.0:
    mat /= mx

np.save(f"fragment_{idx}.npy", mat)
PY
    """
}


process KMEANS {

    tag "kmeans"

    publishDir params.outdir, mode: 'copy'

    input:
    path fragments

    output:
    path "centres.npy"
    path "centres.txt"

    script:
    """
    python3 - <<'PY'
import glob
import time
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import paired_distances

seed        = ${params.seed}
dimensions  = ${params.dimensions}
num_centres = ${params.num_centres}
iterations  = ${params.iterations}
epsilon     = ${params.epsilon}
arity       = ${params.arity}

start_time = time.time()

fragment_files = sorted(glob.glob("fragment_*.npy"))
fragments = [np.load(f) for f in fragment_files]

initialization_time = time.time()
print("Generation/Load done")
print("Starting kmeans")

np.random.seed(seed)

centres = np.asarray([
    np.random.random(dimensions)
    for _ in range(num_centres)
])

old_centres = None
iteration = 0


def partial_sum(fragment, centres):
    partials = np.zeros((centres.shape[0], 2), dtype=object)
    close_centres = pairwise_distances(fragment, centres).argmin(axis=1)

    for center_idx, _ in enumerate(centres):
        indices = np.argwhere(close_centres == center_idx).flatten()
        partials[center_idx][0] = np.sum(fragment[indices], axis=0)
        partials[center_idx][1] = indices.shape[0]

    return partials


def merge(*data):
    accum = data[0].copy()
    for d in data[1:]:
        accum += d
    return accum


def converged(old_centres, centres, epsilon, iteration, max_iter):
    if old_centres is None:
        return False

    dist = np.sum(paired_distances(centres, old_centres))
    return dist < epsilon**2 or iteration >= max_iter


def recompute_centres(partials, old_centres, arity):
    centres = old_centres.copy()

    while len(partials) > 1:
        partials_subset = partials[:arity]
        partials = partials[arity:]
        partials.append(merge(*partials_subset))

    for idx, sum_ in enumerate(partials[0]):
        if sum_[1] != 0:
            centres[idx] = sum_[0] / sum_[1]

    return centres


while not converged(old_centres, centres, epsilon, iteration, iterations):
    print(f"Doing iteration #{iteration + 1}/{iterations}")

    old_centres = centres.copy()

    partials = []
    for frag in fragments:
        partials.append(partial_sum(frag, old_centres))

    centres = recompute_centres(partials, old_centres, arity)
    iteration += 1

kmeans_time = time.time()

print("-----------------------------------------")
print("-------------- RESULTS ------------------")
print("-----------------------------------------")
print("Initialization time: %f" % (initialization_time - start_time))
print("KMeans time: %f" % (kmeans_time - initialization_time))
print("Total time: %f" % (kmeans_time - start_time))
print("-----------------------------------------")
print("CENTRES:")
print(centres)
print("-----------------------------------------")

np.save("centres.npy", centres)
np.savetxt("centres.txt", centres)
PY
    """
}


workflow {

    fragment_infos = []

    points_per_fragment = Math.max(
        1,
        (params.numpoints as int).intdiv(params.fragments as int)
    )

    idx = 0

    (0..<(params.numpoints as int)).step(points_per_fragment) { l ->
        int r = Math.min(params.numpoints as int, l + points_per_fragment)

        fragment_infos << tuple(
            idx,
            r - l,
            params.dimensions as int,
            params.mode,
            params.seed as int + l
        )

        idx = idx + 1
    }

    fragments_ch = Channel.fromList(fragment_infos)

    generated_fragments = GENERATE_FRAGMENT(fragments_ch)

    KMEANS(generated_fragments.collect())
}