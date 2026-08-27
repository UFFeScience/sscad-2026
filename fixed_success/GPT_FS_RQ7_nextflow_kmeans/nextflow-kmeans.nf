nextflow.enable.dsl = 2

/*
 * Parameters equivalent to the PyCOMPSs version
 */
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
    tag "fragment_${fid}"
    publishDir params.outdir, mode: 'copy'

    input:
    tuple val(fid), val(points), val(dim), val(mode), val(seed)

    output:
    tuple val(fid), path("fragment_${fid}.npy")

    script:
    """
    python - <<'PY'
import numpy as np

fid = ${fid}
points = ${points}
dim = ${dim}
mode = "${mode}"
seed = ${seed}

rand = {
    "normal": lambda k: np.random.normal(0, 1, k),
    "uniform": lambda k: np.random.random(k),
}

np.random.seed(seed)
r = rand[mode]

mat = np.asarray([r(dim) for _ in range(points)])

mat -= np.min(mat)
mx = np.max(mat)
if mx > 0.0:
    mat /= mx

np.save(f"fragment_{fid}.npy", mat)
PY
    """
}

process RUN_KMEANS {
    tag "kmeans"
    publishDir params.outdir, mode: 'copy'

    input:
    path fragments
    val dimensions
    val num_centres
    val iterations
    val seed
    val epsilon
    val arity

    output:
    path "centres.npy"
    path "centres.txt"

    script:
    def fragment_names = fragments.collect { it.name }.join(' ')

    """
    python - <<'PY'
import numpy as np

from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import paired_distances

fragment_files = "${fragment_names}".split()

dimensions = int(${dimensions})
num_centres = int(${num_centres})
iterations = int(${iterations})
seed = int(${seed})
epsilon = float(${epsilon})
arity = int(${arity})

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
    return dist < epsilon ** 2 or iteration >= max_iter

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

np.random.seed(seed)

centres = np.asarray([
    np.random.random(dimensions)
    for _ in range(num_centres)
])

fragments = [np.load(f) for f in fragment_files]

old_centres = None
iteration = 0

while not converged(old_centres, centres, epsilon, iteration, iterations):
    print(f"Doing iteration #{iteration + 1}/{iterations}")

    old_centres = centres.copy()

    partials = []
    for frag in fragments:
        partials.append(partial_sum(frag, old_centres))

    centres = recompute_centres(partials, old_centres, arity)
    iteration += 1

np.save("centres.npy", centres)

with open("centres.txt", "w") as f:
    f.write(str(centres))
    f.write("\\n")

print("-----------------------------------------")
print("CENTRES:")
print(centres)
print("-----------------------------------------")
PY
    """
}

workflow {
    log.info """
    K M E A N S   N E X T F L O W
    =============================
    points      : ${params.numpoints}
    dimensions  : ${params.dimensions}
    centres     : ${params.num_centres}
    fragments   : ${params.fragments}
    mode        : ${params.mode}
    iterations  : ${params.iterations}
    epsilon     : ${params.epsilon}
    outdir      : ${params.outdir}
    """.stripIndent()

    def points_per_fragment = Math.max(1, params.numpoints.intdiv(params.fragments))

    def fragment_meta = []
    def fid = 0

    (0..<params.numpoints).step(points_per_fragment) { l ->
        int r = Math.min(params.numpoints, l + points_per_fragment)

        fragment_meta << tuple(
            fid,
            r - l,
            params.dimensions,
            params.mode,
            params.seed + l
        )

        fid = fid + 1
    }

    fragments_ch = GENERATE_FRAGMENT(Channel.fromList(fragment_meta))

    RUN_KMEANS(
        fragments_ch.map { id, frag -> frag }.collect(),
        params.dimensions,
        params.num_centres,
        params.iterations,
        params.seed,
        params.epsilon,
        params.arity
    )
}