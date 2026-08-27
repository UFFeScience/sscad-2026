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


@task(returns=list)
def matmul_block(A, B):
    """
    Multiply two blocks of matrices.
    :param A: numpy array block
    :param B: numpy array block
    :return: product block
    """
    return np.dot(A, B)


@task(returns=list)
def matmul_accumulate(partial_products):
    """
    Sum a list of matrices.
    :param partial_products: list of numpy arrays
    :return: sum of arrays
    """
    result = partial_products[0]
    for mat in partial_products[1:]:
        result = result + mat
    return result


def run_matmul(size=512, block_size=128):
    """
    Orchestrate the matrix multiplication workflow.
    """
    print("Starting Matmul workflow...")
    np.random.seed(0)
    A = np.random.rand(size, size)
    B = np.random.rand(size, size)

    # Split matrices into blocks
    n_blocks = size // block_size
    blocks_A = []
    blocks_B = []
    for i in range(n_blocks):
        row_start = i * block_size
        row_end = row_start + block_size
        for j in range(n_blocks):
            col_start = j * block_size
            col_end = col_start + block_size
            blocks_A.append(A[row_start:row_end, col_start:col_end])
            blocks_B.append(B[row_start:row_end, col_start:col_end])

    # Compute partial products: each block of A multiplied by each block of B
    # For simplicity, we compute the full product block-wise.
    # We'll use a simple approach: for each output block, sum over k.
    # We'll create tasks for each output block.
    output_blocks = []
    for i in range(n_blocks):
        for j in range(n_blocks):
            # For output block (i,j), sum over k of A[i,k] * B[k,j]
            partials = []
            for k in range(n_blocks):
                # Get block A[i,k] and B[k,j]
                A_block = A[i*block_size:(i+1)*block_size, k*block_size:(k+1)*block_size]
                B_block = B[k*block_size:(k+1)*block_size, j*block_size:(j+1)*block_size]
                partials.append(matmul_block(A_block, B_block))
            # NOTE (rerun-simple fix): 'partials' e' uma lista de futures
            # (AppFuture) de matmul_block ainda nao resolvidos. Passa-la
            # diretamente para matmul_accumulate (outra @task) faz o worker
            # tentar iterar/somar objetos AppFuture em vez de arrays reais
            # (TypeError / task failure). Resolvido com compss_wait_on()
            # antes de repassar a lista para a task de reducao.
            partials = compss_wait_on(partials)
            # Accumulate partials
            output_block = matmul_accumulate(partials)
            output_blocks.append(output_block)

    # Wait for all results
    output_blocks = compss_wait_on(output_blocks)

    # Assemble final matrix
    C = np.zeros((size, size))
    idx = 0
    for i in range(n_blocks):
        for j in range(n_blocks):
            C[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = output_blocks[idx]
            idx += 1

    # Verify with numpy (optional)
    C_expected = np.dot(A, B)
    print("Matmul completed. Max error:", np.max(np.abs(C - C_expected)))


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
