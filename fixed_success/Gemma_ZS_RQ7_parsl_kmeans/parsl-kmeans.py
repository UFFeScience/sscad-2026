import numpy as np
import time
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import paired_distances
import parsl
from parsl.app.app import python_app
from parsl.configs.local_threads import config

# Configuração básica para execução local
parsl.load(config)

@python_app
def partial_sum(fragment, centres):
    partials = np.zeros((centres.shape[0], 2), dtype=object)
    close_centres = pairwise_distances(fragment, centres).argmin(axis=1)
    for center_idx, _ in enumerate(centres):
        indices = np.argwhere(close_centres == center_idx).flatten()
        partials[center_idx][0] = np.sum(fragment[indices], axis=0) if indices.size > 0 else np.zeros(fragment.shape[1])
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
    # No Parsl, precisamos resolver os futures para a redução
    partials = [p.result() for p in partials_futures]
    centres = old_centres.copy()
    
    # Redução manual simulando a arity do PyCompss
    while len(partials) > 1:
        next_level = []
        for i in range(0, len(partials), arity):
            chunk = partials[i:i + arity]
            next_level.append(merge(chunk).result())
        partials = next_level
    
    final_partial = partials[0]
    for idx, sum_ in enumerate(final_partial):
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
        # Lança as tarefas de soma parcial em paralelo
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
    
    # Geração paralela de fragmentos
    fragment_futures = []
    for l in range(0, numpoints, points_per_fragment):
        r = min(numpoints, l + points_per_fragment)
        fragment_futures.append(generate_fragment(r - l, dimensions, mode, seed + l))
    
    # Sincronização (equivalente ao compss_barrier)
    fragment_list = [f.result() for f in fragment_futures]
    print("Generation/Load done")
    initialization_time = time.time()
    
    centres = kmeans_frag(fragment_list, dimensions, num_centres, iterations, seed, epsilon, arity)
    kmeans_time = time.time()

    print(f"Initialization time: {initialization_time - start_time:.4f}")
    print(f"KMeans time: {kmeans_time - initialization_time:.4f}")
    print("CENTRES:\n", centres)

if __name__ == "__main__":
    # Simulando parse_arguments para brevidade
    main(0, 100, 2, 5, 10, "uniform", 20, 1e-9, 50)