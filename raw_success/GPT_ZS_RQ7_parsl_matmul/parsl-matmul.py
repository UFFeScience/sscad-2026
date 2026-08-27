import time
import argparse
from concurrent.futures import wait

import numpy as np
import parsl
from parsl import python_app


def ensure_parsl():
    try:
        parsl.dfk()
    except Exception:
        parsl.load()


@python_app
def generate_block(size, num_blocks, seed=0, set_to_zero=False):
    import numpy as np

    np.random.seed(seed)

    if not set_to_zero:
        b = np.random.random((size, size))

        # Normalize matrix to ensure more numerical precision
        b /= np.sum(b) * float(num_blocks)
    else:
        b = np.zeros((size, size))

    return b


@python_app
def fused_multiply_add(A, B, C):
    """
    Versão Parsl do FMA.

    Em PyCOMPSs:
        C += np.dot(A, B)

    Em Parsl:
        retorna uma nova versão de C.
    """
    import numpy as np

    C_new = C.copy()
    C_new += np.dot(A, B)

    return C_new


def dot(A, B, C):
    """
    Blocked matrix multiplication usando Parsl.

    C[i][j] é atualizado criando uma cadeia de dependências:
        C_0 -> C_1 -> C_2 -> ... -> C_final
    """

    n = len(A)
    m = len(B[0])

    for i in range(n):
        for j in range(m):
            for k in range(n):
                C[i][j] = fused_multiply_add(
                    A[i][k],
                    B[k][j],
                    C[i][j],
                )

    return C


def main(num_blocks, elems_per_block, seed):
    ensure_parsl()

    start_time = time.time()

    A, B, C = [], [], []

    for i in range(num_blocks):
        A.append([])
        B.append([])
        C.append([])

        bid = 0

        for j in range(num_blocks):
            A[i].append(
                generate_block(
                    elems_per_block,
                    num_blocks,
                    seed=seed + bid,
                )
            )
            bid += 1

            B[i].append(
                generate_block(
                    elems_per_block,
                    num_blocks,
                    seed=seed + bid,
                )
            )
            bid += 1

            C[i].append(
                generate_block(
                    elems_per_block,
                    num_blocks,
                    set_to_zero=True,
                )
            )

    generated_futures = []

    for matrix in [A, B, C]:
        for row in matrix:
            generated_futures.extend(row)

    # Equivalente ao compss_barrier após a geração
    wait(generated_futures)

    initialization_time = time.time()

    C = dot(A, B, C)

    final_futures = []

    for row in C:
        final_futures.extend(row)

    # Equivalente ao compss_barrier após a multiplicação
    wait(final_futures)

    multiplication_time = time.time()

    print("-----------------------------------------")
    print("-------------- RESULTS ------------------")
    print("-----------------------------------------")
    print("Initialization time: %f" % (initialization_time - start_time))
    print("Multiplication time: %f" % (multiplication_time - initialization_time))
    print("Total time: %f" % (multiplication_time - start_time))
    print("-----------------------------------------")

    # Caso queira materializar a matriz resultado no processo principal:
    # C_result = [[block.result() for block in row] for row in C]
    # print(C_result)


def parse_args():
    description = "Parsl blocked matmul implementation"

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "-b",
        "--num_blocks",
        type=int,
        default=1,
        help="Number of blocks, N in NxN",
    )

    parser.add_argument(
        "-e",
        "--elems_per_block",
        type=int,
        default=2,
        help="Elements per block, N in NxN",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Pseudo-Random seed",
    )

    return parser.parse_args()


if __name__ == "__main__":
    opts = parse_args()
    main(**vars(opts))