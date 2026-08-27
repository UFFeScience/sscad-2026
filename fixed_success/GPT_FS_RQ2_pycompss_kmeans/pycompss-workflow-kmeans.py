import os



import shlex



import time



import shutil



import argparse



import subprocess



import urllib.request



import numpy as np



from pycompss.api.task import task



from pycompss.api.api import compss_wait_on



from pycompss.api.parameter import FILE_IN, FILE_OUT



def reduce_tree(items, reducer):
    """
    Redução em árvore para evitar uma cadeia longa de dependências.
    """
    if not items:
        return None

    while len(items) > 1:
        new_items = []
        for i in range(0, len(items), 2):
            if i + 1 < len(items):
                new_items.append(reducer(items[i], items[i + 1]))
            else:
                new_items.append(items[i])
        items = new_items

    return items[0]



def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path



@task(returns=list)
def generate_points_chunk(seed, n_points, dimensions, true_clusters, std):
    """
    Gera um bloco de pontos sintéticos.
    """
    rng = np.random.default_rng(seed)

    centers = rng.normal(loc=0.0, scale=10.0, size=(true_clusters, dimensions))
    labels = rng.integers(0, true_clusters, size=n_points)

    x = centers[labels] + rng.normal(loc=0.0, scale=std, size=(n_points, dimensions))
    return x.tolist()



@task(returns=dict)
def kmeans_partial(points, centroids):
    """
    Atribui pontos ao centróide mais próximo e retorna somas parciais,
    contagens e inércia parcial.
    """
    x = np.asarray(points, dtype=np.float64)
    c = np.asarray(centroids, dtype=np.float64)

    k = c.shape[0]
    d = c.shape[1]

    distances = ((x[:, None, :] - c[None, :, :]) ** 2).sum(axis=2)
    labels = np.argmin(distances, axis=1)

    sums = np.zeros((k, d), dtype=np.float64)
    counts = np.zeros(k, dtype=np.int64)

    np.add.at(sums, labels, x)
    np.add.at(counts, labels, 1)

    inertia = float(np.sum(distances[np.arange(x.shape[0]), labels]))

    return {
        "sums": sums.tolist(),
        "counts": counts.tolist(),
        "inertia": inertia,
    }



@task(returns=dict, priority=True)
def merge_kmeans_stats(a, b):
    """
    Junta dois acumuladores parciais do KMeans.
    """
    sums_a = np.asarray(a["sums"], dtype=np.float64)
    sums_b = np.asarray(b["sums"], dtype=np.float64)

    counts_a = np.asarray(a["counts"], dtype=np.int64)
    counts_b = np.asarray(b["counts"], dtype=np.int64)

    return {
        "sums": (sums_a + sums_b).tolist(),
        "counts": (counts_a + counts_b).tolist(),
        "inertia": float(a["inertia"] + b["inertia"]),
    }



def workflow_kmeans(args):
    start = time.time()

    n_per_chunk = args.samples // args.chunks
    remainder = args.samples % args.chunks

    chunks = []
    for i in range(args.chunks):
        n_i = n_per_chunk + (1 if i < remainder else 0)
        chunks.append(
            generate_points_chunk(
                args.seed + i,
                n_i,
                args.dimensions,
                args.true_clusters,
                args.std,
            )
        )

    rng = np.random.default_rng(args.seed)
    centroids = rng.normal(
        loc=0.0,
        scale=10.0,
        size=(args.k, args.dimensions),
    ).tolist()

    last_inertia = None

    for it in range(args.iterations):
        partials = []
        for chunk in chunks:
            partials.append(kmeans_partial(chunk, centroids))

        merged = reduce_tree(partials, merge_kmeans_stats)
        stats = compss_wait_on(merged)

        sums = np.asarray(stats["sums"], dtype=np.float64)
        counts = np.asarray(stats["counts"], dtype=np.int64)

        old_centroids = np.asarray(centroids, dtype=np.float64)
        new_centroids = old_centroids.copy()

        for j in range(args.k):
            if counts[j] > 0:
                new_centroids[j, :] = sums[j, :] / counts[j]

        shift = float(np.linalg.norm(new_centroids - old_centroids))
        centroids = new_centroids.tolist()

        inertia = stats["inertia"]
        print(
            "Iteração {:03d} | inertia={:.6f} | shift={:.8f}".format(
                it + 1,
                inertia,
                shift,
            )
        )

        if last_inertia is not None and abs(last_inertia - inertia) < args.tol:
            print("Convergência por variação de inertia menor que tol.")
            break

        if shift < args.tol:
            print("Convergência por deslocamento dos centróides menor que tol.")
            break

        last_inertia = inertia

    elapsed = time.time() - start

    print("\nResultado KMeans")
    print("Centróides:")
    for i, c in enumerate(centroids):
        print("  C{} = {}".format(i, c))
    print("Tempo total: {:.6f} s".format(elapsed))



def parse_args():
    parser = argparse.ArgumentParser(
        description="Workflows PyCOMPSs: KMeans, Matmul e Lysozyme/GROMACS"
    )

    subparsers = parser.add_subparsers(dest="workflow", required=True)

    # -------------------------
    # KMeans
    # -------------------------
    p_kmeans = subparsers.add_parser("kmeans", help="Executa workflow KMeans")
    p_kmeans.add_argument("--samples", type=int, default=100000)
    p_kmeans.add_argument("--dimensions", type=int, default=2)
    p_kmeans.add_argument("-k", type=int, default=4)
    p_kmeans.add_argument("--true-clusters", type=int, default=4)
    p_kmeans.add_argument("--chunks", type=int, default=8)
    p_kmeans.add_argument("--iterations", type=int, default=20)
    p_kmeans.add_argument("--tol", type=float, default=1e-4)
    p_kmeans.add_argument("--std", type=float, default=0.7)
    p_kmeans.add_argument("--seed", type=int, default=12345)
    p_kmeans.set_defaults(func=workflow_kmeans)

    # Matmul e Lysozyme nao estao implementados neste arquivo (cada
    # algoritmo foi gerado em um arquivo pycompss-workflow-<algo>.py
    # separado) -- os subparsers correspondentes foram removidos daqui
    # porque referenciavam funcoes (workflow_matmul/workflow_lysozyme)
    # inexistentes neste modulo, causando NameError em parse_args()
    # mesmo ao rodar apenas o subcomando "kmeans".

    return parser.parse_args()



def main():
    args = parse_args()
    args.func(args)



if __name__ == "__main__":
    main()
