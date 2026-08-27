#!/usr/bin/env python3

import numpy as np
import time

import parsl
from parsl.app.app import python_app
from parsl.configs.local_threads import config

parsl.load(config)



@python_app
def generate_block(
        size,
        num_blocks,
        seed=0,
        zero=False):

    import numpy as np

    np.random.seed(seed)


    if zero:

        return np.zeros(
            (size,size)
        )

    b=np.random.random(
        (size,size)
    )

    b /= (
        np.sum(b)
        *
        float(num_blocks)
    )

    return b



@python_app
def fused_multiply_add(A,B,C):

    import numpy as np

    return C + np.dot(A,B)



def dot(A,B,C):

    futures=[]


    n=len(A)

    for i in range(n):

        for j in range(n):

            c=C[i][j]


            for k in range(n):

                futures.append(
                    fused_multiply_add(
                        A[i][k],
                        B[k][j],
                        c
                    )
                )


    return futures



def main():

    start=time.time()

    blocks=2
    size=4


    A=[]
    B=[]
    C=[]


    seed=0


    for i in range(blocks):

        A.append([])
        B.append([])
        C.append([])


        for j in range(blocks):

            A[i].append(
                generate_block(
                    size,
                    blocks,
                    seed
                )
            )

            B[i].append(
                generate_block(
                    size,
                    blocks,
                    seed+1
                )
            )

            C[i].append(
                generate_block(
                    size,
                    blocks,
                    zero=True
                )
            )

            seed+=2



    results=dot(A,B,C)


    final=[
        r.result()
        for r in results
    ]


    print(
        "Blocks computed:",
        len(final)
    )


    print(
        "Time:",
        time.time()-start
    )


if __name__=="__main__":
    main()