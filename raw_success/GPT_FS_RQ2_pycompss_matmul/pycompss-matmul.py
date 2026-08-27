#!/usr/bin/env python3

import argparse
import time
import numpy as np

from pycompss.api.task import task
from pycompss.api.api import compss_wait_on


@task(returns=np.ndarray)
def multiply_block(A, B):

    return np.matmul(A, B)


def main(size, blocks):

    np.random.seed(0)

    A = np.random.rand(size, size)
    B = np.random.rand(size, size)

    A_blocks = np.array_split(A, blocks)

    futures = []

    start = time.time()

    for block in A_blocks:
        futures.append(multiply_block(block, B))

    result = [compss_wait_on(f) for f in futures]

    C = np.vstack(result)

    elapsed = time.time() - start

    print("Matrix shape:", C.shape)
    print("Execution time:", elapsed)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--size", type=int, default=2000)
    parser.add_argument("--blocks", type=int, default=4)

    args = parser.parse_args()

    main(args.size, args.blocks)