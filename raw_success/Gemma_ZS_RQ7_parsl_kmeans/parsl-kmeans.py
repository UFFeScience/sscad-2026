import numpy as np
import time
import argparse
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import paired_distances
from parsl.app.app import python_app
from parsl.configs.local_threads import config 
import parsl

parsl.load(config)

@python_app
def partial_sum(fragment, centres):
    partials = np.zeros((centres.shape[0], 2), dtype=object)
    close_centres = pairwise_distances(fragment, centres).argmin(axis=1)
    for center_idx, _ in enumerate(centres):
        indices = np.argwhere(close_centres == center_idx).flatten()
        partials[center_idx][0] = np.sum(fragment[indices], axis=0) if indices.size > 0 else np.zeros(centres.shape[1])
        partials[center_idx][1] = indices.shape[0]
    return partials

@python_app
def merge(data_list):
    accum = data_list[0].copy()
    for d in data_list[1:]:
        accum += d
    return accum

def converged(old_centres, centres, epsilon, iteration, max_iter):
    if old_centres is None: return False
    dist = np.sum(paired_distances(centres, old_centres))
    return dist < epsilon**2 or iteration >= max_iter

def recompute_centres(partials_futures, old_centres, arity):
    # No Parsl, precisamos resolver os futures para manipular a lista
    partials = [f.result() for f in partials_futures]
    centres = old_centres.copy()
    
    # Simulação da redução por aridade
    while len(partials) > 1:
        next_level = []
        for i in range(0, len(partials), arity):
            subset = partials[i : i + arity]
            if len(subset) > 1:
                next_level.append(merge(subset))
            else:
                next_level.append(subset[0])
        partials = [f.result() if hasattr(f, 'result') else f for f in next_level]
    
    final_partials = partials[0]
    for idx, sum_ in enumerate(final_partials):
        if sum_[1] != 0:
            centres[idx] = sum_[0] / sum_[1]
    return centres

def kmeans_frag(fragments, dimensions, num_centres=10, iterations=20, seed=0.0, epsilon=1e-9, arity=50):
    np.random.seed(seed)
    centres = np.asarray([np.random.random(dimensions) for _ in range(num_centres)])
    old_centres = None
    iteration = 0
    while not converged(old_centres, centres, epsilon, iteration, iterations):
        print(f"Doing iteration #{iteration + 1}/{iterations}")
        old_centres = centres.copy()
        # Lançamento assíncrono das tarefas
        partials_futures = [partial_sum(frag, old_centres) for frag in fragments]
        centres = recompute_centres(partials_futures, old_centres, arity)
        iteration += 1
    return centres

@python_app
def generate_fragment(points, dim, mode, seed):
    rand = {"normal": lambda k: np.random.normal(0, 1, k), "uniform": lambda k: np.random.random(k)}
    r = rand[mode]
    np.random.seed(seed)
    mat = np.asarray([r(dim) for __ in range(points)])
    mat -= np.min(mat)
    mx = np.max(mat)
    if mx > 0.0: mat /= mx
    return mat

def main(seed, numpoints, dimensions, num_centres, fragments, mode, iterations, epsilon, arity):
    start_time = time.time()
    points_per_fragment = max(1, numpoints // fragments)
    
    fragment_futures = []
    for l in range(0, numpoints, points_per_fragment):
        r = min(numpoints, l + points_per_fragment)
        fragment_futures.append(generate_fragment(r - l, dimensions, mode, seed + l))

    # Espera a geração de todos os fragmentos (equivalente ao barrier)
    fragment_list = [f.result() for f in fragment_futures]
    print("Generation/Load done")
    initialization_time = time.time()

    centres = kmeans_frag(fragment_list, dimensions, num_centres, iterations, seed, epsilon, arity)
    kmeans_time = time.time()

    print(f"Initialization time: {initialization_time - start_time:.4f}")
    print(f"KMeans time: {kmeans_time - initialization_time:.4f}")
    print(f"Total time: {kmeans_time - start_time:.4f}")
    print("CENTRES:\n", centres)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("-n", "--numpoints", type=int, default=100)
    parser.add_argument("-d", "--dimensions", type=int, default=2)
    parser.add_argument("-c", "--num_centres", type=int, default=5)
    parser.add_argument("-f", "--fragments", type=int, default=10)
    parser.add_argument("-m", "--mode", type=str, default="uniform")
    parser.add_argument("-i", "--iterations", type=int, default=20)
    parser.add_argument("-e", "--epsilon", type=float, default=1e-9)
    parser.add_argument("-a", "--arity", type=int, default=50)
    args = parser.parse_args()
    main(**vars(args))