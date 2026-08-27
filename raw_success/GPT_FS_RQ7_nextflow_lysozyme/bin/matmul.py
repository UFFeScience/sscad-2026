#!/usr/bin/env python3

import argparse
import numpy as np
import os



# ---------------------------------------------------------
# Generate matrix block
# ---------------------------------------------------------

def generate_block(
        size,
        num_blocks,
        seed,
        zero,
        output
):


    np.random.seed(seed)


    if zero:

        block=np.zeros(
            (size,size)
        )


    else:

        block=np.random.random(
            (size,size)
        )


        block /= (
            np.sum(block)
            *
            float(num_blocks)
        )



    np.save(
        output,
        block
    )



# ---------------------------------------------------------
# Generate complete blocked matrices
# ---------------------------------------------------------

def generate_dataset(
        num_blocks,
        block_size,
        seed,
        output
):


    os.makedirs(
        output,
        exist_ok=True
    )


    files=[]

    bid=0


    for matrix in ["A","B","C"]:


        for i in range(num_blocks):

            for j in range(num_blocks):


                filename=os.path.join(
                    output,
                    f"{matrix}_{i}_{j}.npy"
                )


                generate_block(
                    block_size,
                    num_blocks,
                    seed+bid,
                    matrix=="C",
                    filename
                )


                files.append(filename)

                bid+=1



    return files



# ---------------------------------------------------------
# Block matrix multiplication
# ---------------------------------------------------------

def multiply(
        blocks,
        num_blocks,
        output
):


    matrices={}


    for file in blocks:

        name=os.path.basename(file)

        key=name.replace(".npy","")

        matrices[key]=np.load(file)



    C={}



    for i in range(num_blocks):

        for j in range(num_blocks):


            result=None


            for k in range(num_blocks):


                A=matrices[
                    f"A_{i}_{k}"
                ]


                B=matrices[
                    f"B_{k}_{j}"
                ]



                product=A @ B


                if result is None:

                    result=product

                else:

                    result+=product



            C[
                f"C_{i}_{j}"
            ]=result



    np.save(
        output,
        C
    )



# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

def main():


    parser=argparse.ArgumentParser(
        description="Blocked matrix multiplication"
    )


    parser.add_argument(
        "--generate",
        action="store_true"
    )


    parser.add_argument(
        "--multiply",
        action="store_true"
    )


    parser.add_argument(
        "--size",
        type=int,
        default=2
    )


    parser.add_argument(
        "--blocks",
        nargs="+"
    )


    parser.add_argument(
        "--num_blocks",
        type=int,
        default=2
    )


    parser.add_argument(
        "--seed",
        type=int,
        default=0
    )


    parser.add_argument(
        "--zero",
        action="store_true"
    )


    parser.add_argument(
        "--output",
        default="result.npy"
    )


    args=parser.parse_args()



    if args.generate:


        generate_block(
            args.size,
            args.num_blocks,
            args.seed,
            args.zero,
            args.output
        )


    elif args.multiply:


        multiply(
            args.blocks,
            args.num_blocks,
            args.output
        )



if __name__=="__main__":
    main()