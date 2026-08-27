import json
import os
import time

import numpy as np

from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import paired_distances

from pycompss.api.task import task
from pycompss.api.api import compss_wait_on
from pycompss.api.api import compss_barrier


# --- instrumentation added for makespan / cpu / parallelism-efficiency analysis ---
# Timing is recorded INSIDE each @task function body so it measures real execution
# wall time inside the COMPSs worker sandbox, not call/submission overhead on the master.
def _record(name, t0, t1):
    # NOTE: COMPSs launches workers over SSH to localhost, which does not propagate the
    # parent shell's environment -- os.environ.get("WF_METRICS_FILE", ...) is invisible
    # inside the worker, so the metrics path is hardcoded to the container-shared mount.
    mf = "/workspace/kmeans/metrics.jsonl"
    line = json.dumps({"task": name, "start": t0, "end": t1, "duration": t1 - t0})
    with open(mf, "a") as f:
        f.write(line + "\n")
# --- end instrumentation ---


@task(returns=np.ndarray)
def partial_sum(fragment, centres):
    _t0 = time.time()
    partials = np.zeros((centres.shape[0], 2), dtype=object)
    close_centres = pairwise_distances(fragment, centres).argmin(axis=1)
    for center_idx, _ in enumerate(centres):
        indices = np.argwhere(close_centres == center_idx).flatten()
        partials[center_idx][0] = np.sum(fragment[indices], axis=0)
        partials[center_idx][1] = indices.shape[0]
    _record("partial_sum", _t0, time.time())
    return partials


@task(returns=dict)
def merge(*data):
    _t0 = time.time()
    accum = data[0].copy()
    for d in data[1:]:
        accum += d
    _record("merge", _t0, time.time())
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
    partials = compss_wait_on(partials)
    for idx, sum_ in enumerate(partials[0]):
        if sum_[1] != 0:
            centres[idx] = sum_[0] / sum_[1]
    return centres


def kmeans_frag(
    fragments,
    dimensions,
    num_centres=10,
    iterations=20,
    seed=0.0,
    epsilon=1e-9,
    arity=50,
):
    np.random.seed(seed)
    centres = np.asarray([np.random.random(dimensions) for _ in range(num_centres)])
    old_centres = None
    iteration = 0
    while not converged(old_centres, centres, epsilon, iteration, iterations):
        print("Doing iteration #%d/%d" % (iteration + 1, iterations))
        old_centres = centres.copy()
        partials = []
        for frag in fragments:
            partial = partial_sum(frag, old_centres)
            partials.append(partial)
        centres = recompute_centres(partials, old_centres, arity)
        iteration += 1
    return centres


def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser(description="KMeans Clustering.")
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("-n", "--numpoints", type=int, default=100)
    parser.add_argument("-d", "--dimensions", type=int, default=2)
    parser.add_argument("-c", "--num_centres", type=int, default=5)
    parser.add_argument("-f", "--fragments", type=int, default=10)
    parser.add_argument("-m", "--mode", type=str, default="uniform", choices=["uniform", "normal"])
    parser.add_argument("-i", "--iterations", type=int, default=20)
    parser.add_argument("-e", "--epsilon", type=float, default=1e-9)
    parser.add_argument("-a", "--arity", type=int, default=50)
    return parser.parse_args()


@task(returns=1)
def generate_fragment(points, dim, mode, seed):
    _t0 = time.time()
    rand = {
        "normal": lambda k: np.random.normal(0, 1, k),
        "uniform": lambda k: np.random.random(k),
    }
    r = rand[mode]
    np.random.seed(seed)
    mat = np.asarray([r(dim) for __ in range(points)])
    mat -= np.min(mat)
    mx = np.max(mat)
    if mx > 0.0:
        mat /= mx
    _record("generate_fragment", _t0, time.time())
    return mat


def main(
    seed,
    numpoints,
    dimensions,
    num_centres,
    fragments,
    mode,
    iterations,
    epsilon,
    arity,
):
    start_time = time.time()

    fragment_list = []
    points_per_fragment = max(1, numpoints // fragments)

    for l in range(0, numpoints, points_per_fragment):
        r = min(numpoints, l + points_per_fragment)
        fragment_list.append(generate_fragment(r - l, dimensions, mode, seed + l))

    compss_barrier()
    print("Generation/Load done")
    initialization_time = time.time()
    print("Starting kmeans")

    centres = kmeans_frag(
        fragments=fragment_list,
        dimensions=dimensions,
        num_centres=num_centres,
        iterations=iterations,
        seed=seed,
        epsilon=epsilon,
        arity=arity,
    )
    compss_barrier()
    print("Ending kmeans")
    kmeans_time = time.time()

    print("-----------------------------------------")
    print("-------------- RESULTS ------------------")
    print("-----------------------------------------")
    print("Initialization time: %f" % (initialization_time - start_time))
    print("KMeans time: %f" % (kmeans_time - initialization_time))
    print("Total time: %f" % (kmeans_time - start_time))
    print("-----------------------------------------")
    centres = compss_wait_on(centres)
    print("CENTRES:")
    print(centres)
    print("-----------------------------------------")


if __name__ == "__main__":
    options = parse_arguments()
    main(**vars(options))
