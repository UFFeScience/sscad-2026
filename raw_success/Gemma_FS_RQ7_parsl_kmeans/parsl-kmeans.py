import numpy as np
import time
import parsl
from parsl.app.app import python_app
from parsl.configs.local_threads import config
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import paired_distances

parsl.load(config)

@python_app
def partial_sum(fragment, centres):
    import numpy as np
    from sklearn.metrics import pairwise_distances
    partials = np.zeros((centres.shape[0], 2), dtype=object)
    close_centres = pairwise_distances(fragment, centres).argmin(axis=1)
    for center_idx, _ in enumerate(centres):
        indices = np.argwhere(close_centres == center_idx).flatten()
        partials[center_idx][0] = np.sum(fragment[indices], axis=0) if indices.size > 0 else np.zeros(fragment.shape[1])
        partials[center_idx][1] = indices.shape[0]
    return partials

@python_app
def merge(*data):
    import numpy as np
    accum = data[0].copy()
    for d in data[1:]:
        accum += d
    return accum

@python_app
def generate_fragment(points, dim, mode, seed):
    import numpy as np
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
    if old_centres is None:
        return False
    dist = np.sum(paired_distances(centres, old_centres))
    return dist < epsilon**2 or iteration >= max_iter

def recompute_centres(partials_futures, old_centres, arity):
    # No Parsl, precisamos resolver as futures para realizar a redução
    partials = [p.result() for p in partials_futures]
    
    while len(partials) > 1:
        partials_subset = partials[:arity]
        partials = partials[arity:]
        # O merge é disparado como app e aguardamos o resultado
        partials.append(merge(*partials_subset).result())
    
    centres = old_centres.copy()
    for idx, sum_ in enumerate(partials[0]):
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
        partials_futures = []
        for frag in fragments:
            # frag aqui é a future do generate_fragment
            partials_futures.append(partial_sum(frag, old_centres))
        centres = recompute_centres(partials_futures, old_centres, arity)
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

def main(**kwargs):
    start_time = time.time()
    points_per_fragment = max(1, kwargs['numpoints'] // kwargs['fragments'])
    
    fragment_list = []
    for l in range(0, kwargs['numpoints'], points_per_fragment):
        r = min(kwargs['numpoints'], l + points_per_fragment)
        fragment_list.append(generate_fragment(r - l, kwargs['dimensions'], kwargs['mode'], kwargs['seed'] + l))

    # Aguarda a geração dos fragmentos (equivalente ao compss_barrier)
    [f.result() for f in fragment_list]
    print("Generation/Load done")
    initialization_time = time.time()
    
    centres = kmeans_frag(
        fragments=fragment_list,
        dimensions=kwargs['dimensions'],
        num_centres=kwargs['num_centres'],
        iterations=kwargs['iterations'],
        seed=kwargs['seed'],
        epsilon=kwargs['epsilon'],
        arity=kwargs['arity']
    )
    
    kmeans_time = time.time()
    print(f"Initialization time: {initialization_time - start_time}")
    print(f"KMeans time: {kmeans_time - initialization_time}")
    print(f"Total time: {kmeans_time - start_time}")
    print("CENTRES:\n", centres)

if __name__ == "__main__":
    options = parse_arguments()
    main(**vars(options))