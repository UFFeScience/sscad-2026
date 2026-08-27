#!/usr/bin/env python3
"""
Parsl translation of the PyCOMPSs KMeans clustering script.
"""

import os
import argparse
import time
import numpy as np

import parsl
from parsl.app.app import python_app
from parsl.configs.local_threads import config

# Load the local threads configuration
parsl.load(config)

# Import required sklearn functions
from sklearn.metrics.pairwise import pairwise_distances, paired_distances


@python_app
def partial_sum(fragment, centres):
    """
    Compute partial sums for each centre.
    Returns a numpy array of shape (num_centres, 2) where column 0 is sum of points,
    column 1 is count.
    """
    # Compute distances from each point to all centres
    dists = pairwise_distances(fragment, centres)  # shape (n_points, num_centres)
    close_centres = np.argmin(dists, axis=1)      # shape (n_points,)

    num_centres = centres.shape[0]
    partials = np.zeros((num_centres, 2), dtype=object)

    for center_idx in range(num_centres):
        indices = np.where(close_centres == center_idx)[0]
        if indices.size > 0:
            partials[center_idx, 0] = np.sum(fragment[indices], axis=0)
            partials[center_idx, 1] = indices.size
        else:
            # No point assigned to this centre; store zero sum and zero count
            partials[center_idx, 0] = np.zeros(fragment.shape[1])
            partials[center_idx, 1] = 0

    return partials


@python_app
def merge_multi(*data):
    """
    Merge multiple partial-sum arrays by element‑wise summation.
    data : variable number of numpy arrays of shape (num_centres, 2)
    Returns a single numpy array of the same shape.
    """
    # Sum along the first axis (stacking the arrays)
    return np.sum(data, axis=0)


def converged(old_centres, centres, epsilon, iteration, max_iter):
    """Check for convergence of the K‑Means algorithm."""
    if old_centres is None:
        return False
    dist = np.sum(paired_distances(centres, old_centres))
    return dist < epsilon**2 or iteration >= max_iter


def recompute_centres(partials_results, old_centres, arity):
    """
    Aggregate partial results and compute new centres.
    partials_results : list of numpy arrays (output of partial_sum)
    old_centres      : current centres (used only for shape information)
    arity            : reduction arity (not used in this simplified version)
    Returns a new numpy array of centres.
    """
    # Merge all partial arrays using the parallel merge_multi task
    merged = merge_multi(*partials_results).result()
    num_centres = merged.shape[0]
    new_centres = np.empty((num_centres, old_centres.shape[1]))

    for idx in range(num_centres):
        sum_ = merged[idx]
        if sum_[1] != 0:
            new_centres[idx] = sum_[0] / sum_[1]
        else:
            # Keep the old centre if no point was assigned
            new_centres[idx] = old_centres[idx]

    return new_centres


def kmeans_main(fragments, dimensions, num_centres=10, iterations=20,
                seed=0.0, epsilon=1e-9, arity=50):
    """
    Core K‑Means loop using Parsl tasks for distance computation.
    """
    np.random.seed(seed)
    centres = np.asarray([np.random.random(dimensions) for _ in range(num_centres)])
    old_centres = None

    for iteration in range(iterations):
        print(f"Doing iteration #{iteration + 1}/{iterations}")
        old_centres = centres.copy()

        # Submit partial_sum tasks for every fragment
        partial_futures = [partial_sum(frag, old_centres) for frag in fragments]

        # Gather results
        partials_results = [f.result() for f in partial_futures]

        # Compute new centres
        centres = recompute_centres(partials_results, old_centres, arity)

        # Check convergence
        if converged(old_centres, centres, epsilon, iteration + 1, iterations):
            break

    return centres


@python_app
def generate_fragment(points, dim, mode, seed):
    """
    Generate a random fragment (numpy array) with the given parameters.
    This function is executed as a Parsl task; it returns the array directly.
    """
    np.random.seed(seed)
    if mode == 'normal':
        mat = np.random.normal(0, 1, (points, dim))
    elif mode == 'uniform':
        mat = np.random.random((points, dim))
    else:
        raise ValueError("mode must be 'normal' or 'uniform'")

    # Normalise to [0, 1]
    mat -= np.min(mat)
    mx = np.max(mat)
    if mx > 0.0:
        mat /= mx
    return mat


def parse_args():
    """Parse command‑line arguments."""
    parser = argparse.ArgumentParser(description="KMeans Clustering (Parsl version).")
    parser.add_argument("-s", "--seed", type=int, default=0,
                        help="Pseudo‑random seed. Default = 0")
    parser.add_argument("-n", "--numpoints", type=int, default=100,
                        help="Number of points. Default = 100")
    parser.add_argument("-d", "--dimensions", type=int, default=2,
                        help="Number of dimensions. Default = 2")
    parser.add_argument("-c", "--num_centres", type=int, default=5,
                        help="Number of centres. Default = 5")
    parser.add_argument("-f", "--fragments", type=int, default=10,
                        help="Number of fragments. Default = 10 (fragments < points)")
    parser.add_argument("-m", "--mode", type=str, default="uniform",
                        choices=["uniform", "normal"],
                        help="Distribution of points. Default = uniform")
    parser.add_argument("-i", "--iterations", type=int, default=20,
                        help="Maximum number of iterations")
    parser.add_argument("-e", "--epsilon", type=float, default=1e-9,
                        help="Epsilon. KMeans stops when |old - new| < epsilon.")
    parser.add_argument("-a", "--arity", type=int, default=50,
                        help="Arity of the reduction for centroid recomputation")
    return parser.parse_args()


def main(seed, numpoints, dimensions, num_centres, fragments, mode,
         iterations, epsilon, arity):
    """Main driver: generate fragments, run K‑Means, and report timing."""
    start_time = time.time()

    # --- Generate fragments -------------------------------------------------
    fragment_futures = []
    points_per_fragment = max(1, numpoints // fragments)
    for l in range(0, numpoints, points_per_fragment):
        r = min(numpoints, l + points_per_fragment)
        fragment_futures.append(generate_fragment(r - l, dimensions, mode, seed + l))

    # Wait for all fragment generation tasks
    fragments_data = [f.result() for f in fragment_futures]
    print("Generation/Load done")

    # --- Run K‑Means --------------------------------------------------------
    initialization_time = time.time()
    print("Starting kmeans")
    centres = kmeans_main(fragments_data, dimensions, num_centres,
                          iterations, seed, epsilon, arity)
    print("Ending kmeans")
    kmeans_time = time.time()

    # --- Output results -----------------------------------------------------
    print("-----------------------------------------")
    print("-------------- RESULTS ------------------")
    print("-----------------------------------------")
    print(f"Initialization time: {initialization_time - start_time:.6f}")
    print(f"KMeans time: {kmeans_time - initialization_time:.6f}")
    print(f"Total time: {kmeans_time - start_time:.6f}")
    print("-----------------------------------------")
    print("CENTRES:")
    print(centres)
    print("-----------------------------------------")


if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))