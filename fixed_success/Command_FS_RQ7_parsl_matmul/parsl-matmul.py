#!/usr/bin/env python3
"""
Parsl translation of the PyCOMPSs blocked matrix multiplication script.
"""

import os
import argparse
import time
import numpy as np

import parsl
from parsl.app.app import python_app
from parsl.configs.local_threads import config

parsl.load(config)


@python_app
def generate_block(size, num_blocks, seed=0, set_to_zero=False):
    """
    Generate a square block of the given size.
    Returns a numpy array of shape (size, size).
    """
    np.random.seed(seed)
    if not set_to_zero:
        b = np.random.random((size, size))
        # Normalise to improve numerical precision
        b /= np.sum(b) * float(num_blocks)
    else:
        b = np.zeros((size, size))
    return b


@python_app
def fused_multiply_add(A_block, B_block):
    """
    Compute the matrix product of two blocks.
    Returns a numpy array of shape (size, size).
    """
    return np.dot(A_block, B_block)


def main(num_blocks, elems_per_block, seed):
    """
    Driver for blocked matrix multiplication using Parsl tasks.
    """
    start_time = time.time()

    # --- Generate matrix blocks --------------------------------------------
    # A and B are lists of lists of blocks; C will be a list of lists of zero blocks
    A_futures = []
    B_futures = []
    C_futures = []

    bid = 0
    for i in range(num_blocks):
        A_row = []
        B_row = []
        C_row = []
        for _ in range(num_blocks):
            # Generate A[i][k] and B[k][j] with different seeds
            A_block = generate_block(elems_per_block, num_blocks, seed=seed + bid)
            B_block = generate_block(elems_per_block, num_blocks, seed=seed + bid + 1)
            A_row.append(A_block)
            B_row.append(B_block)
            bid += 1
        # C blocks are initialised to zero
        for _ in range(num_blocks):
            C_block = generate_block(elems_per_block, num_blocks, seed=seed + bid, set_to_zero=True)
            C_row.append(C_block)
            bid += 1
        A_futures.append(A_row)
        B_futures.append(B_row)
        C_futures.append(C_row)

    # Wait for all block generation tasks
    A = [[b.result() for b in row] for row in A_futures]
    B = [[b.result() for b in row] for row in B_futures]
    C = [[b.result() for b in row] for row in C_futures]

    initialization_time = time.time()
    print("Initialization done")

    # --- Compute all block products in parallel -----------------------------
    # Submit fused_multiply_add for every (i, j, k) combination
    mult_futures = []
    for i in range(num_blocks):
        for j in range(num_blocks):
            for k in range(num_blocks):
                mult_futures.append(fused_multiply_add(A[i][k], B[k][j]))

    # Wait for all products
    products = [f.result() for f in mult_futures]

    # Accumulate products into the corresponding C[i][j] blocks
    mult_idx = 0
    for i in range(num_blocks):
        for j in range(num_blocks):
            C[i][j] += products[mult_idx]
            mult_idx += 1

    multiplication_time = time.time()
    print("Multiplication done")

    # --- Output timing -------------------------------------------------------
    print("-----------------------------------------")
    print("-------------- RESULTS ------------------")
    print("-----------------------------------------")
    print(f"Initialization time: {initialization_time - start_time:.6f}")
    print(f"Multiplication time: {multiplication_time - initialization_time:.6f}")
    print(f"Total time: {multiplication_time - start_time:.6f}")
    print("-----------------------------------------")
    # For demonstration, print a few entries of the result
    print("First row of C (first 5 entries):")
    print(C[0][0][:5])
    print("-----------------------------------------")


def parse_args():
    """Parse command‑line arguments."""
    parser = argparse.ArgumentParser(description="Blocked matrix multiplication (Parsl version).")
    parser.add_argument("-b", "--num_blocks", type=int, default=1,
                        help="Number of blocks (N in NxN)")
    parser.add_argument("-e", "--elems_per_block", type=int, default=2,
                        help="Elements per block (size of each block)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Pseudo‑random seed")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))