import numpy as np
import time

import parsl
from parsl import python_app
from parsl.data_provider.files import File

from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import paired_distances


# -----------------------------
# Parsl configuration
# -----------------------------

from parsl.config import Config
from parsl.executors import ThreadPoolExecutor

config = Config(
    executors=[
        ThreadPoolExecutor(
            max_threads=8
        )
    ]
)

parsl.load(config)



# -----------------------------
# Tasks
# -----------------------------


@python_app
def partial_sum(fragment, centres):

    partials = np.zeros(
        (centres.shape[0], 2),
        dtype=object
    )

    close_centres = pairwise_distances(
        fragment,
        centres
    ).argmin(axis=1)


    for center_idx, _ in enumerate(centres):

        indices = np.argwhere(
            close_centres == center_idx
        ).flatten()

        partials[center_idx][0] = np.sum(
            fragment[indices],
            axis=0
        )

        partials[center_idx][1] = indices.shape[0]


    return partials



@python_app
def merge(*data):

    accum = data[0].copy()

    for d in data[1:]:
        accum += d

    return accum



@python_app
def generate_fragment(points, dim, mode, seed):

    rand = {
        "normal": lambda k:
            np.random.normal(0,1,k),

        "uniform":
            lambda k:
            np.random.random(k)
    }


    np.random.seed(seed)

    r = rand[mode]

    mat = np.asarray(
        [
            r(dim)
            for _ in range(points)
        ]
    )


    mat -= np.min(mat)

    mx=np.max(mat)

    if mx>0:
        mat /= mx


    return mat



# -----------------------------
# KMeans
# -----------------------------


def converged(
        old_centres,
        centres,
        epsilon,
        iteration,
        max_iter):


    if old_centres is None:
        return False


    dist=np.sum(
        paired_distances(
            centres,
            old_centres
        )
    )


    return (
        dist < epsilon**2
        or iteration >= max_iter
    )



def recompute_centres(
        partials,
        old_centres,
        arity):


    centres = old_centres.copy()


    while len(partials)>1:

        subset = partials[:arity]

        partials = partials[arity:]

        partials.append(
            merge(
                *subset
            )
        )


    partials = [
        p.result()
        for p in partials
    ]


    for idx,sum_ in enumerate(partials[0]):

        if sum_[1]!=0:
            centres[idx]=sum_[0]/sum_[1]


    return centres



def kmeans_frag(
        fragments,
        dimensions,
        num_centres=10,
        iterations=20,
        seed=0,
        epsilon=1e-9,
        arity=50):


    np.random.seed(seed)


    centres=np.asarray(
        [
            np.random.random(dimensions)
            for _ in range(num_centres)
        ]
    )


    old_centres=None

    iteration=0



    while not converged(
        old_centres,
        centres,
        epsilon,
        iteration,
        iterations):


        old_centres=centres.copy()


        partials=[]


        for frag in fragments:

            partials.append(
                partial_sum(
                    frag,
                    old_centres
                )
            )


        centres=recompute_centres(
            partials,
            old_centres,
            arity
        )


        iteration+=1


    return centres



# -----------------------------
# Main
# -----------------------------


def main():

    start=time.time()


    fragments=[]


    for i in range(10):

        fragments.append(
            generate_fragment(
                100,
                2,
                "uniform",
                i
            )
        )


    fragments=[
        f.result()
        for f in fragments
    ]


    centres=kmeans_frag(
        fragments,
        2
    )


    print(
        centres
    )


    print(
        "time:",
        time.time()-start
    )



if __name__=="__main__":

    main()