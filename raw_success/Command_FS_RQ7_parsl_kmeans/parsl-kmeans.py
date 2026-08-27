#!/usr/bin/env python3
"""
Translated from PyCOMPSs KMeans script.
Uses Python apps for computation and a reduction pattern to mimic the original task graph.
"""

import os
import argparse
import numpy as np
import time

import parsl
from parsl.app.app import python_app, bash_app
from parsl.configs.local_threads import config

# Load a local thread configuration (adjust as needed)
parsl.load(config)

# ----------------------------------------------------------------------
# Helper: paired distances for convergence check (sklearn import kept for fidelity)
from sklearn.metrics.pairwise import paired_distances
# ----------------------------------------------------------------------

@python_app
def generate_fragment(points, dim, mode, seed):
    """
    Generate a random fragment of `points` points with `dim` dimensions.
    Returns a NumPy array of shape (points, dim).
    """
    rand_funcs = {
        "normal": lambda k: np.random.normal(0, 1, k),
        "uniform": lambda k: np.random.random(k),
    }
    r = rand_funcs[mode]
    np.random.seed(seed)
    mat = np.asarray([r(dim) for _ in range(points)])
    # Normalize to [0, 1]
    mat -= np.min(mat)
    mx = np.max(mat)
    if mx > 0.0:
        mat /= mx
    return mat


@python_app
def partial_sum(fragment, centres):
    """
    Compute per‑centre sum and count of points assigned to each centre.
    Returns an array of shape (n_centres, 2) where each row is [sum, count].
    """
    n_centres = centres.shape[0]
    # Use broadcasting to compute squared Euclidean distances
    diff = fragment[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists = np.sum(diff ** 2, axis=2)
    close_centres = np.argmin(dists, axis=1)

    partials = np.zeros((n_centres, 2), dtype=object)
    for center_idx in range(n_centres):
        indices = np.argwhere(close_centres == center_idx).flatten()
        if indices.size > 0:
            partials[center_idx][0] = np.sum(fragment[indices], axis=0)
            partials[center_idx][1] = indices.shape[0]
        else:
            # No points assigned – keep zero sum and count
            partials[center_idx][0] = np.zeros(fragment.shape[1])
            partials[center_idx][1] = 0
    return partials


@python_app
def merge_arrays(*arrays):
    """
    Element‑wise sum of multiple partials arrays.
    Each array has shape (n_centres, 2). Returns their sum.
    """
    total = arrays[0].copy()
    for arr in arrays[1:]:
        total += arr
    return total


def reduce_partials(futures, arity):
    """
    Recursively merge a list of futures using `merge_arrays` with given arity.
    Returns a single future representing the final sum.
    """
    while len(futures) > 1:
        new_futures = []
        for i in range(0, len(futures), arity):
            chunk = futures[i:i + arity]
            if len(chunk) == 1:
                new_futures.append(chunk[0])
            else:
                merged_future = merge_arrays(*chunk)
                new_futures.append(merged_future)
        futures = new_futures
    return futures[0]


def kmeans_frag(fragments, dimensions, num_centres, iterations, seed, epsilon, arity):
    """
    Core K‑Means algorithm using Parsl tasks for partial sums and reduction.
    """
    np.random.seed(seed)
    centres = np.random.random((num_centres, dimensions))
    old_centres = None
    iteration = 0

    while True:
        if iteration > 0:
            print(f"Doing iteration #{iteration}/{iterations}")

        # Convergence test (same as original)
        if old_centres is not None:
            dist = np.sum(paired_distances(centres, old_centres))
            if dist < epsilon ** 2 or iteration >= iterations:
                break

        old_centres = centres.copy()

        # Compute partial sums for every fragment (parallel tasks)
        partial_futures = [partial_sum(frag, old_centres) for frag in fragments]

        # Reduce all partials using the arity parameter
        total_partials_future = reduce_partials(partial_futures, arity)
        total_partials = total_partials_future.result()

        # Recompute centres
        centres = np.empty((num_centres, dimensions))
        for idx, (sum_, count) in enumerate(total_partials):
            if count != 0:
                centres[idx] = sum_ / count
            else:
                centres[idx] = old_centres[idx]   # keep previous centre

        iteration += 1

    return centres


def main(seed, numpoints, dimensions, num_centres, fragments, mode, iterations, epsilon, arity):
    start_time = time.time()

    # ---- Generate all fragments as separate tasks ----
    fragment_futures = []
    points_per_fragment = max(1, numpoints // fragments)
    for l in range(0, numpoints, points_per_fragment):
        r = min(numpoints, l + points_per_fragment)
        frag_future = generate_fragment(r - l, dimensions, mode, seed + l)
        fragment_futures.append(frag_future)

    # Wait for fragment generation
    fragments_data = [f.result() for f in fragment_futures]
    print("Generation/Load done")

    # ---- Run K‑Means ----
    init_time = time.time()
    print("Starting kmeans")
    centres = kmeans_frag(fragments_data, dimensions, num_centres,
                          iterations, seed, epsilon, arity)
    end_time = time.time()

    # ---- Output results ----
    print("-----------------------------------------")
    print("-------------- RESULTS ------------------")
    print("-----------------------------------------")
    print(f"Initialization time: {init_time - start_time:.6f}")
    print(f"KMeans time: {end_time - init_time:.6f}")
    print(f"Total time: {end_time - start_time:.6f}")
    print("-----------------------------------------")
    print("CENTRES:")
    print(centres)
    print("-----------------------------------------")


def parse_arguments():
    parser = argparse.ArgumentParser(description="KMeans Clustering (Parsl)")
    parser.add_argument("-s", "--seed", type=int, default=0,
                        help="Pseudo-random seed. Default = 0")
    parser.add_argument("-n", "--numpoints", type=int, default=100,
                        help="Number of points. Default = 100")
    parser.add_argument("-d", "--dimensions", type=int, default=2,
                        help="Number of dimensions. Default = 2")
    parser.add_argument("-c", "--num_centres", type=int, default=5,
                        help="Number of centres. Default = 5")
    parser.add_argument("-f", "--fragments", type=int, default=10,
                        help="Number of fragments. Default = 10")
    parser.add_argument("-m", "--mode", type=str, default="uniform",
                        choices=["uniform", "normal"],
                        help="Distribution of points. Default = uniform")
    parser.add_argument("-i", "--iterations", type=int, default=20,
                        help="Maximum number of iterations")
    parser.add_argument("-e", "--epsilon", type=float, default=1e-9,
                        help="Epsilon. KMeans will stop when |old - new| < epsilon.")
    parser.add_argument("-a", "--arity", type=int, default=50,
                        help="Arity of the reduction carried out during the computation of the new centroids")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(**vars(args))