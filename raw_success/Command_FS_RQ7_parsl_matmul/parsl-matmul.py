#!/usr/bin/env python3
"""
Translated from PyCOMPSs blocked matrix multiplication script.
Demonstrates task‑based block generation and parallel computation of the product.
"""

import os
import argparse
import numpy as np
import time

import parsl
from parsl.app.app import python_app
from parsl.configs.local_threads import config

parsl.load(config)


@python_app
def generate_block(size, num_blocks, seed=0, set_to_zero=False):
    """
    Generate a square block of given size.
    Returns a NumPy array of shape (size, size).
    """
    np.random.seed(seed)
    if not set_to_zero:
        b = np.random.random((size, size))
        # Normalize as in the original
        b /= np.sum(b) * float(num_blocks)
    else:
        b = np.zeros((size, size))
    return b


@python_app
def compute_C_block(i, j, A_blocks, B_blocks, elems_per_block, num_blocks):
    """
    Compute the (i,j) block of C = A × B by summing over k.
    A_blocks and B_blocks are 2‑D lists of NumPy arrays.
    Returns the computed block as a NumPy array.
    """
    total = np.zeros((elems_per_block, elems_per_block))
    for k in range(num_blocks):
        total += np.dot(A_blocks[i][k], B_blocks[k][j])
    return total


def main(num_blocks, elems_per_block, seed):
    start_total = time.time()

    # ---- Generate all A and B blocks ----
    # Store futures temporarily to later retrieve values
    A_futures = [[None] * num_blocks for _ in range(num_blocks)]
    B_futures = [[None] * num_blocks for _ in range(num_blocks)]
    all_gen_futures = []

    for i in range(num_blocks):
        for k in range(num_blocks):
            fut = generate_block(elems_per_block, num_blocks,
                                 seed=seed + i * num_blocks + k)
            A_futures[i][k] = fut
            all_gen_futures.append(fut)

    for k in range(num_blocks):
        for j in range(num_blocks):
            fut = generate_block(elems_per_block, num_blocks,
                                 seed=seed + (k + 1) * num_blocks + j)
            B_futures[k][j] = fut
            all_gen_futures.append(fut)

    # Wait for all block generation tasks
    for f in all_gen_futures:
        f.result()

    # Retrieve the actual arrays
    A_blocks = [[None] * num_blocks for _ in range(num_blocks)]
    B_blocks = [[None] * num_blocks for _ in range(num_blocks)]
    for i in range(num_blocks):
        for k in range(num_blocks):
            A_blocks[i][k] = A_futures[i][k].result()
    for k in range(num_blocks):
        for j in range(num_blocks):
            B_blocks[k][j] = B_futures[k][j].result()

    print("Block generation done")
    init_time = time.time()

    # ---- Compute matrix multiplication in parallel ----
    comp_futures = []
    for i in range(num_blocks):
        for j in range(num_blocks):
            fut = compute_C_block(i, j, A_blocks, B_blocks,
                                  elems_per_block, num_blocks)
            comp_futures.append(fut)

    # Wait for all computation tasks
    for f in comp_futures:
        f.result()

    mult_time = time.time()
    print("Multiplication done")

    # ---- Output timing information ----
    print("-----------------------------------------")
    print("-------------- RESULTS ------------------")
    print("-----------------------------------------")
    print(f"Initialization time: {init_time - start_total:.6f}")
    print(f"Multiplication time: {mult_time - init_time:.6f}")
    print(f"Total time: {mult_time - start_total:.6f}")
    print("-----------------------------------------")
    # Optionally print a summary of the result (e.g., sum of all entries)
    C_full = np.zeros((num_blocks * elems_per_block, num_blocks * elems_per_block))
    for i in range(num_blocks):
        for j in range(num_blocks):
            C_full[i * elems_per_block:(i + 1) * elems_per_block,
                   j * elems_per_block:(j + 1) * elems_per_block] = comp_futures[i * num_blocks + j].result()
    print(f"Sum of all entries in C: {np.sum(C_full):.6f}")
    print("-----------------------------------------")


def parse_args():
    parser = argparse.ArgumentParser(description="COMPSs blocked matmul implementation (Parsl)")
    parser.add_argument('-b', '--num_blocks', type=int, default=1,
                        help='Number of blocks (N in NxN)')
    parser.add_argument('-e', '--elems_per_block', type=int, default=2,
                        help='Elements per block (N in NxN)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Pseudo-Random seed')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.num_blocks, args.elems_per_block, args.seed)