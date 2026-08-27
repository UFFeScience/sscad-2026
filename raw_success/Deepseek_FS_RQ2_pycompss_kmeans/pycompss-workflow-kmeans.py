"""
PyCOMPSs workflows for KMeans, Matmul, and Lysozyme.

This script demonstrates how to define and run separate workflows for
three different algorithms using the PyCOMPSs framework. Each algorithm
is implemented as a set of tasks and can be invoked independently via
command-line arguments.

Usage examples:
    runcompss script.py --algorithm kmeans
    runcompss script.py --algorithm matmul
    runcompss script.py --algorithm lysozyme
"""



import os



import sys



import time



import argparse



import subprocess



import numpy as np



from pycompss.api.task import task



from pycompss.api.api import compss_wait_on



@task(returns=dict)
def kmeans_assign(points, centroids):
    """
    Assign each point to the nearest centroid.
    :param points: numpy array of shape (n_samples, n_features)
    :param centroids: numpy array of shape (n_clusters, n_features)
    :return: dictionary with cluster_id -> list of point indices
    """
    n_samples = points.shape[0]
    assignments = {}
    for i in range(n_samples):
        dist = np.linalg.norm(points[i] - centroids, axis=1)
        cluster = int(np.argmin(dist))
        assignments.setdefault(cluster, []).append(i)
    return assignments



@task(returns=list)
def kmeans_update(points, assignments, n_clusters):
    """
    Update centroids based on current assignments.
    :param points: numpy array of shape (n_samples, n_features)
    :param assignments: dictionary cluster_id -> list of indices
    :param n_clusters: total number of clusters
    :return: list of new centroid coordinates
    """
    new_centroids = []
    for c in range(n_clusters):
        idx = assignments.get(c, [])
        if idx:
            new_centroids.append(np.mean(points[idx], axis=0))
        else:
            # keep previous centroid if no points assigned
            new_centroids.append(np.zeros(points.shape[1]))
    return new_centroids



@task(returns=dict)
def kmeans_merge_assignments(dict1, dict2):
    """
    Merge two assignment dictionaries.
    :param dict1: first dictionary
    :param dict2: second dictionary
    :return: merged dictionary
    """
    merged = dict1.copy()
    for k, v in dict2.items():
        merged.setdefault(k, []).extend(v)
    return merged



def run_kmeans(n_samples=1000, n_features=2, n_clusters=3, max_iter=10):
    """
    Orchestrate the KMeans workflow.
    """
    print("Starting KMeans workflow...")
    # Generate synthetic data
    np.random.seed(42)
    data = np.random.rand(n_samples, n_features)

    # Split data into chunks (e.g., 4 chunks)
    n_chunks = 4
    chunk_size = n_samples // n_chunks
    chunks = []
    for i in range(n_chunks):
        start = i * chunk_size
        end = start + chunk_size if i < n_chunks - 1 else n_samples
        chunks.append(data[start:end])

    # Initialize centroids
    centroids = data[:n_clusters]

    for iteration in range(max_iter):
        # Parallel assignment: each chunk computes partial assignments
        partial_assignments = []
        for chunk in chunks:
            partial_assignments.append(kmeans_assign(chunk, centroids))

        # Merge partial assignments
        full_assignments = {}
        for partial in partial_assignments:
            full_assignments = kmeans_merge_assignments(full_assignments, partial)
        full_assignments = compss_wait_on(full_assignments)

        # Update centroids (single task, could be parallelized further)
        new_centroids = kmeans_update(data, full_assignments, n_clusters)
        new_centroids = compss_wait_on(new_centroids)
        centroids = np.array(new_centroids)

    print("KMeans completed. Final centroids:")
    print(centroids)



def parse_args():
    parser = argparse.ArgumentParser(description="PyCOMPSs workflows for KMeans, Matmul, and Lysozyme")
    parser.add_argument("--algorithm", choices=["kmeans", "matmul", "lysozyme"], required=True,
                        help="Which algorithm to run")
    # KMeans args
    parser.add_argument("--samples", type=int, default=1000, help="KMeans: number of samples")
    parser.add_argument("--features", type=int, default=2, help="KMeans: number of features")
    parser.add_argument("--clusters", type=int, default=3, help="KMeans: number of clusters")
    parser.add_argument("--max-iter", type=int, default=10, help="KMeans: max iterations")
    # Matmul args
    parser.add_argument("--size", type=int, default=512, help="Matmul: matrix size")
    parser.add_argument("--block-size", type=int, default=128, help="Matmul: block size")
    # Lysozyme args
    parser.add_argument("--pdb", type=str, help="Lysozyme: input PDB file")
    parser.add_argument("--mdp", type=str, help="Lysozyme: MDP file")
    parser.add_argument("--output-dir", type=str, default="lysozyme_out", help="Lysozyme: output directory")
    parser.add_argument("--nsteps", type=int, default=1000, help="Lysozyme: number of MD steps")
    parser.add_argument("--use-mpi", action="store_true", help="Lysozyme: use gmx_mpi instead of gmx")
    return parser.parse_args()



def main():
    args = parse_args()
    if args.algorithm == "kmeans":
        run_kmeans(n_samples=args.samples, n_features=args.features,
                   n_clusters=args.clusters, max_iter=args.max_iter)
    elif args.algorithm == "matmul":
        run_matmul(size=args.size, block_size=args.block_size)
    elif args.algorithm == "lysozyme":
        if not args.pdb or not args.mdp:
            print("Error: Lysozyme requires --pdb and --mdp arguments.")
            sys.exit(1)
        run_lysozyme(pdb_file=args.pdb, mdp_file=args.mdp,
                     output_dir=args.output_dir, nsteps=args.nsteps,
                     use_mpi=args.use_mpi)



if __name__ == "__main__":
    main()
