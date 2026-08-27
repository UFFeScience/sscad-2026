import time
import numpy as np

from parsl import python_app
import parsl
from parsl.config import Config
from parsl.executors.threads import ThreadPoolExecutor

parsl.load(Config(executors=[ThreadPoolExecutor(max_threads=4)]))


@python_app
def generate_block(size, num_blocks, seed=0, set_to_zero=False):
    np.random.seed(seed)
    if not set_to_zero:
        b = np.random.random((size, size))
        b /= np.sum(b) * float(num_blocks)
    else:
        b = np.zeros((size, size))
    return b


@python_app
def fused_multiply_add(A, B, C):
    # No PyCOMPSs, C era INOUT. No Parsl, retornamos o C atualizado.
    return C + np.dot(A, B)


def dot(A, B, C):
    n = len(A)
    m = len(B[0])

    for i in range(n):
        for j in range(m):
            for k in range(n):
                C[i][j] = fused_multiply_add(A[i][k], B[k][j], C[i][j])


def main(num_blocks, elems_per_block, seed):
    start_time = time.time()

    A, B, C = [], [], []

    for i in range(num_blocks):
        A.append([])
        B.append([])
        C.append([])

        bid = 0
        for j in range(num_blocks):
            A[-1].append(generate_block(elems_per_block, num_blocks, seed=seed + bid))
            bid += 1

            B[-1].append(generate_block(elems_per_block, num_blocks, seed=seed + bid))
            bid += 1

            C[-1].append(generate_block(elems_per_block, num_blocks, set_to_zero=True))

    # compss_barrier -> espera a geração dos blocos
    for matrix in (A, B, C):
        for row in matrix:
            for blk in row:
                blk.result()

    initialization_time = time.time()

    dot(A, B, C)

    parsl.wait_for_current_tasks()
    multiplication_time = time.time()

    print("-----------------------------------------")
    print("-------------- RESULTS ------------------")
    print("-----------------------------------------")
    print("Initialization time: %f" % (initialization_time - start_time))
    print("Multiplication time: %f" % (multiplication_time - initialization_time))
    print("Total time: %f" % (multiplication_time - start_time))
    print("-----------------------------------------")


def parse_args():
    import argparse

    description = "Parsl blocked matmul implementation"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "-b", "--num_blocks", type=int, default=1,
        help="Number of blocks (N in NxN)"
    )
    parser.add_argument(
        "-e", "--elems_per_block", type=int, default=2,
        help="Elements per block (N in NxN)"
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Pseudo-Random seed"
    )
    return parser.parse_args()


if __name__ == "__main__":
    opts = parse_args()
    main(**vars(opts))