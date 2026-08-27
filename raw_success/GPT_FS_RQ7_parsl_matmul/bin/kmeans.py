#!/usr/bin/env python3

import argparse
import numpy as np
import os

from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import paired_distances



# ---------------------------------------------------------
# Generate random fragment
# ---------------------------------------------------------

def generate_fragment(points, dim, mode, seed, output):

    np.random.seed(seed)

    if mode == "normal":
        generator = lambda k: np.random.normal(0, 1, k)

    else:
        generator = lambda k: np.random.random(k)


    matrix = np.asarray(
        [
            generator(dim)
            for _ in range(points)
        ]
    )


    # Normalize
    matrix -= np.min(matrix)

    maximum = np.max(matrix)

    if maximum > 0:
        matrix /= maximum


    np.save(output, matrix)



# ---------------------------------------------------------
# Partial KMeans computation
# ---------------------------------------------------------

def partial_sum(fragment, centres):

    partials = np.zeros(
        (centres.shape[0], 2),
        dtype=object
    )


    closest = pairwise_distances(
        fragment,
        centres
    ).argmin(axis=1)


    for idx in range(len(centres)):

        points = np.argwhere(
            closest == idx
        ).flatten()


        partials[idx][0] = np.sum(
            fragment[points],
            axis=0
        )


        partials[idx][1] = len(points)


    return partials



# ---------------------------------------------------------
# Merge partial results
# ---------------------------------------------------------

def merge(partials):

    result = partials[0].copy()


    for p in partials[1:]:

        for i in range(len(result)):

            result[i][0] += p[i][0]
            result[i][1] += p[i][1]


    return result



# ---------------------------------------------------------
# KMeans algorithm
# ---------------------------------------------------------

def kmeans(
        fragments,
        dimensions,
        num_centres,
        iterations,
        seed,
        epsilon
):


    np.random.seed(seed)


    centres = np.asarray(
        [
            np.random.random(dimensions)
            for _ in range(num_centres)
        ]
    )


    old_centres = None


    for iteration in range(iterations):


        if old_centres is not None:

            distance = np.sum(
                paired_distances(
                    centres,
                    old_centres
                )
            )


            if distance < epsilon**2:
                break



        print(
            f"Iteration {iteration+1}/{iterations}"
        )


        old_centres = centres.copy()


        partials=[]


        for fragment_file in fragments:


            fragment = np.load(fragment_file)


            partials.append(
                partial_sum(
                    fragment,
                    old_centres
                )
            )



        merged = merge(partials)



        for idx,data in enumerate(merged):

            if data[1] != 0:

                centres[idx] = (
                    data[0] / data[1]
                )



    return centres



# ---------------------------------------------------------
# Generate complete dataset
# ---------------------------------------------------------

def generate_dataset(
        numpoints,
        fragments,
        dimensions,
        mode,
        seed,
        output
):


    os.makedirs(output, exist_ok=True)


    points_per_fragment = max(
        1,
        numpoints // fragments
    )


    files=[]


    for i,start in enumerate(
        range(
            0,
            numpoints,
            points_per_fragment
        )
    ):


        size=min(
            points_per_fragment,
            numpoints-start
        )


        filename=os.path.join(
            output,
            f"fragment_{i}.npy"
        )


        generate_fragment(
            size,
            dimensions,
            mode,
            seed+i,
            filename
        )


        files.append(filename)


    return files



# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

def main():

    parser=argparse.ArgumentParser(
        description="KMeans workflow"
    )


    parser.add_argument(
        "--generate",
        action="store_true"
    )


    parser.add_argument(
        "--run",
        action="store_true"
    )


    parser.add_argument(
        "--points",
        type=int,
        default=100
    )


    parser.add_argument(
        "--dim",
        type=int,
        default=2
    )


    parser.add_argument(
        "--fragments",
        type=int,
        default=10
    )


    parser.add_argument(
        "--mode",
        default="uniform"
    )


    parser.add_argument(
        "--seed",
        type=int,
        default=0
    )


    parser.add_argument(
        "--fragment-files",
        nargs="+"
    )


    parser.add_argument(
        "--centres",
        type=int,
        default=5
    )


    parser.add_argument(
        "--dimensions",
        type=int,
        default=2
    )


    parser.add_argument(
        "--iterations",
        type=int,
        default=20
    )


    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-9
    )


    parser.add_argument(
        "--output",
        default="centres.npy"
    )


    args=parser.parse_args()



    if args.generate:


        generate_fragment(
            args.points,
            args.dim,
            args.mode,
            args.seed,
            args.output
        )


    elif args.run:


        centres=kmeans(
            args.fragment_files,
            args.dimensions,
            args.centres,
            args.iterations,
            args.seed,
            args.epsilon
        )


        np.save(
            args.output,
            centres
        )



if __name__=="__main__":
    main()