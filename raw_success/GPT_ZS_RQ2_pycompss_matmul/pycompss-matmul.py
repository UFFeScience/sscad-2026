import numpy as np
import time

from pycompss.api.task import task
from pycompss.api.api import compss_wait_on
from pycompss.api.api import compss_barrier
from pycompss.api.parameter import IN, OUT


# ============================================================
# MATRIX MULTIPLICATION WORKFLOW
# ============================================================


@task(returns=np.ndarray)
def multiply_block(A, B):

    return np.matmul(A, B)



def matmul_workflow(A, B, block_size=100):

    print("\nExecutando Matmul")


    n = A.shape[0]

    tasks = []

    result = np.zeros(
        (n,n)
    )


    for i in range(0,n,block_size):

        for j in range(0,n,block_size):

            partial = []

            for k in range(0,n,block_size):

                block_A = A[
                    i:i+block_size,
                    k:k+block_size
                ]

                block_B = B[
                    k:k+block_size,
                    j:j+block_size
                ]


                partial.append(
                    multiply_block(
                        block_A,
                        block_B
                    )
                )


            tasks.append(
                (i,j,partial)
            )


    for i,j,partial in tasks:

        blocks = compss_wait_on(partial)

        result[
            i:i+block_size,
            j:j+block_size
        ] = sum(blocks)



    print("Matmul finalizado")

    return result


# ============================================================
# MAIN WORKFLOW
# ============================================================


def main():

    # -------------------------------
    # Matmul
    # -------------------------------


    A = np.random.rand(
        500,
        500
    )

    B = np.random.rand(
        500,
        500
    )


    C = matmul_workflow(
        A,
        B
    )


    print(
        "Resultado Matmul:",
        C.shape
    )


    compss_barrier()


if __name__ == "__main__":

    main()
