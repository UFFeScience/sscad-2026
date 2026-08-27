import argparse
import json
import os
import threading
import time
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import paired_distances

import parsl
from parsl import python_app
from parsl.config import Config
from parsl.executors import ThreadPoolExecutor

config = Config(executors=[ThreadPoolExecutor(max_threads=4)])
parsl.load(config)

# --- instrumentation added for makespan / cpu / parallelism-efficiency analysis ---
_METRICS_FILE = os.environ.get("WF_METRICS_FILE", "metrics.jsonl")
_METRICS_LOCK = threading.Lock()


def _record(name, t0, t1):
    line = json.dumps({"task": name, "start": t0, "end": t1, "duration": t1 - t0})
    with _METRICS_LOCK:
        with open(_METRICS_FILE, "a") as f:
            f.write(line + "\n")


def _timed(name):
    def deco(fn):
        def wrapper(*args, **kwargs):
            t0 = time.time()
            try:
                return fn(*args, **kwargs)
            finally:
                _record(name, t0, time.time())
        wrapper.__name__ = fn.__name__
        return wrapper
    return deco
# --- end instrumentation ---


@python_app
@_timed("generate_fragment")
def generate_fragment(points, dim, mode, seed):
    import numpy as np
    rand = {
        'normal': lambda k: np.random.normal(0, 1, k),
        'uniform': lambda k: np.random.random(k),
    }
    np.random.seed(seed)
    mat = np.asarray([rand[mode](dim) for _ in range(points)])
    mat -= np.min(mat)
    mx = np.max(mat)
    if mx > 0.0:
        mat /= mx
    return mat


@python_app
@_timed("partial_sum")
def partial_sum(fragment, centres):
    import numpy as np
    from sklearn.metrics import pairwise_distances
    partials = np.zeros((centres.shape[0], 2), dtype=object)
    close_centres = pairwise_distances(fragment, centres).argmin(axis=1)
    for center_idx in range(len(centres)):
        indices = np.argwhere(close_centres == center_idx).flatten()
        partials[center_idx][0] = np.sum(fragment[indices], axis=0)
        partials[center_idx][1] = indices.shape[0]
    return partials


@python_app
@_timed("merge_partials")
def merge_partials(*data):
    import numpy as np
    accum = data[0].copy()
    for d in data[1:]:
        accum += d
    return accum


def converged(old_centres, centres, epsilon, iteration, max_iter):
    if old_centres is None:
        return False
    dist = np.sum(paired_distances(centres, old_centres))
    return dist < epsilon ** 2 or iteration >= max_iter


def kmeans_frag(fragments, dimensions, num_centres=10, iterations=20, seed=0.0, epsilon=1e-9):
    np.random.seed(seed)
    centres = np.asarray([np.random.random(dimensions) for _ in range(num_centres)])
    old_centres = None
    iteration = 0
    while not converged(old_centres, centres, epsilon, iteration, iterations):
        print("Doing iteration #%d/%d" % (iteration + 1, iterations))
        old_centres = centres.copy()
        partials = [partial_sum(frag, old_centres) for frag in fragments]
        merged = merge_partials(*[p.result() for p in partials]).result()
        for idx, sum_ in enumerate(merged):
            if sum_[1] != 0:
                centres[idx] = sum_[0] / sum_[1]
        iteration += 1
    return centres


def main(seed, numpoints, dimensions, num_centres, fragments, mode, iterations, epsilon, arity):
    start_time = time.time()

    points_per_fragment = max(1, numpoints // fragments)
    fragment_list = []
    for l in range(0, numpoints, points_per_fragment):
        r = min(numpoints, l + points_per_fragment)
        fragment_list.append(generate_fragment(r - l, dimensions, mode, seed + l))

    fragment_list = [f.result() for f in fragment_list]
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
    )

    print("Ending kmeans")
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


def parse_arguments():
    parser = argparse.ArgumentParser(description='KMeans Clustering.')
    parser.add_argument('-s', '--seed', type=int, default=0)
    parser.add_argument('-n', '--numpoints', type=int, default=100)
    parser.add_argument('-d', '--dimensions', type=int, default=2)
    parser.add_argument('-c', '--num_centres', type=int, default=5)
    parser.add_argument('-f', '--fragments', type=int, default=10)
    parser.add_argument('-m', '--mode', type=str, default='uniform', choices=['uniform', 'normal'])
    parser.add_argument('-i', '--iterations', type=int, default=20)
    parser.add_argument('-e', '--epsilon', type=float, default=1e-9)
    parser.add_argument('-a', '--arity', type=int, default=50)
    return parser.parse_args()


if __name__ == '__main__':
    options = parse_arguments()
    main(**vars(options))
