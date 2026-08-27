#!/usr/bin/env python3
import time
import numpy as np
import parsl
from parsl.app.app import python_app
from parsl.configs.local_threads import config

parsl.load(config)

@python_app
def partial_sum(fragment, centres):
    """Calcula somas parciais para cada centroide."""
    from sklearn.metrics import pairwise_distances
    partials = np.zeros((centres.shape[0], 2), dtype=object)
    close_centres = pairwise_distances(fragment, centres).argmin(axis=1)
    for center_idx, _ in enumerate(centres):
        indices = np.argwhere(close_centres == center_idx).flatten()
        partials[center_idx][0] = np.sum(fragment[indices], axis=0)
        partials[center_idx][1] = indices.shape[0]
    return partials

@python_app
def merge(*data):
    """Acumula múltiplos resultados parciais."""
    accum = data[0].copy()
    for d in data[1:]:
        accum += d
    return accum

@python_app
def generate_fragment(points, dim, mode, seed):
    """Gera um fragmento de dados distribuído."""
    rand = {
        "normal": lambda k: np.random.normal(0, 1, k),
        "uniform": lambda k: np.random.random(k),
    }
    r = rand[mode]
    np.random.seed(seed)
    mat = np.asarray([r(dim) for __ in range(points)])
    mat -= np.min(mat)
    mx = np.max(mat)
    if mx > 0.0:
        mat /= mx
    return mat

def converged(old_centres, centres, epsilon, iteration, max_iter):
    """Critério de convergência."""
    from sklearn.metrics.pairwise import paired_distances
    if old_centres is None:
        return False
    dist = np.sum(paired_distances(centres, old_centres))
    return dist < epsilon**2 or iteration >= max_iter

def recompute_centres(partials, old_centres, arity):
    """Reduz os resultados parciais e recalcula os centróides."""
    centres = old_centres.copy()
    # Redução em árvore
    while len(partials) > 1:
        subset = partials[:arity]
        partials = partials[arity:]
        partials.append(merge(*subset))
    # Espera o resultado final
    final_partials = partials[0].result()
    for idx, sum_ in enumerate(final_partials):
        if sum_[1] != 0:
            centres[idx] = sum_[0] / sum_[1]
    return centres

def kmeans_frag(fragments, dimensions, num_centres=10, iterations=20,
                seed=0.0, epsilon=1e-9, arity=50):
    """Algoritmo K-means baseado em fragmentos."""
    np.random.seed(seed)
    centres = np.asarray([np.random.random(dimensions) for _ in range(num_centres)])
    old_centres = None
    iteration = 0
    while not converged(old_centres, centres, epsilon, iteration, iterations):
        print("Doing iteration #%d/%d" % (iteration + 1, iterations))
        old_centres = centres.copy()
        partials = []
        for frag in fragments:
            partial = partial_sum(frag, old_centres)
            partials.append(partial)
        centres = recompute_centres(partials, old_centres, arity)
        iteration += 1
    return centres

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(description="KMeans Clustering.")
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("-n", "--numpoints", type=int, default=100)
    parser.add_argument("-d", "--dimensions", type=int, default=2)
    parser.add_argument("-c", "--num_centres", type=int, default=5)
    parser.add_argument("-f", "--fragments", type=int, default=10)
    parser.add_argument("-m", "--mode", type=str, default="uniform", choices=["uniform", "normal"])
    parser.add_argument("-i", "--iterations", type=int, default=20)
    parser.add_argument("-e", "--epsilon", type=float, default=1e-9)
    parser.add_argument("-a", "--arity", type=int, default=50)
    return parser.parse_args()

def main(seed, numpoints, dimensions, num_centres, fragments, mode, iterations, epsilon, arity):
    start_time = time.time()
    fragment_list = []
    points_per_fragment = max(1, numpoints // fragments)
    for l in range(0, numpoints, points_per_fragment):
        r = min(numpoints, l + points_per_fragment)
        fragment_list.append(generate_fragment(r - l, dimensions, mode, seed + l))
    # Espera a geração de todos os fragmentos
    [f.result() for f in fragment_list]
    print("Generation/Load done")
    initialization_time = time.time()
    print("Starting kmeans")
    centres = kmeans_frag(fragments=fragment_list, dimensions=dimensions,
                          num_centres=num_centres, iterations=iterations,
                          seed=seed, epsilon=epsilon, arity=arity)
    # O resultado de kmeans_frag já é um array (após .result() interno)
    print("Ending kmeans")
    kmeans_time = time.time()
    print("-----------------------------------------")
    print("-------------- RESULTS ------------------")
    print("-----------------------------------------")
    print("Initialization time: %f" % (initialization_time - start_time))
    print("KMeans time: %f" % (kmeans_time - initialization_time))
    print("Total time: %f" % (kmeans_time - start_time))
    print("-----------------------------------------")
    print("CENTRES:")
    print(centres)
    print("-----------------------------------------")

if __name__ == "__main__":
    options = parse_arguments()
    main(**vars(options))
    parsl.clear()