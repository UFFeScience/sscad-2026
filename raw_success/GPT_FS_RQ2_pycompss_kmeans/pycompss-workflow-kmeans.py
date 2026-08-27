import os



import time



import argparse



import subprocess



import numpy as np



from pycompss.api.task import task



from pycompss.api.api import compss_wait_on



from pycompss.api.parameter import FILE_IN, FILE_OUT, FILE_INOUT



def tree_reduce(items, reducer_task):
    """
    Redução em árvore usando uma task PyCOMPSs.
    """
    if not items:
        return None

    while len(items) > 1:
        new_items = []
        for i in range(0, len(items), 2):
            if i + 1 < len(items):
                new_items.append(reducer_task(items[i], items[i + 1]))
            else:
                new_items.append(items[i])
        items = new_items

    return items[0]



@task(returns=object)
def km_generate_points(n_points, dim, seed, n_clusters):
    """
    Gera pontos sintéticos para o KMeans.
    Os dados são gerados em torno de centros aleatórios.
    """
    rng = np.random.default_rng(seed)

    centers = rng.normal(loc=0.0, scale=10.0, size=(n_clusters, dim))
    labels = rng.integers(0, n_clusters, size=n_points)

    noise = rng.normal(loc=0.0, scale=1.0, size=(n_points, dim))
    points = centers[labels] + noise

    return points



@task(returns=tuple)
def km_partial_assignment(points, centroids):
    """
    Associa cada ponto ao centroide mais próximo e calcula somas parciais.
    """
    k = centroids.shape[0]

    distances = ((points[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    labels = np.argmin(distances, axis=1)

    sums = np.zeros_like(centroids)
    counts = np.zeros(k, dtype=np.int64)

    np.add.at(sums, labels, points)
    counts += np.bincount(labels, minlength=k)

    inertia = float(np.min(distances, axis=1).sum())

    return sums, counts, inertia



@task(returns=tuple, priority=True)
def km_reduce_stats(a, b):
    """
    Junta estatísticas parciais do KMeans.
    """
    sums_a, counts_a, inertia_a = a
    sums_b, counts_b, inertia_b = b

    return sums_a + sums_b, counts_a + counts_b, inertia_a + inertia_b



@task(returns=object)
def km_update_centroids(stats, old_centroids):
    """
    Atualiza centroides a partir das somas e contagens globais.
    """
    sums, counts, _ = stats

    new_centroids = old_centroids.copy()

    for i in range(len(counts)):
        if counts[i] > 0:
            new_centroids[i] = sums[i] / counts[i]

    return new_centroids



def run_kmeans(args):
    """
    Workflow PyCOMPSs para KMeans.
    """
    start = time.time()

    n_points = args.points
    dim = args.dim
    k = args.clusters
    partitions = args.partitions
    iterations = args.iterations
    seed = args.seed

    print("Executando workflow KMeans")
    print("Pontos:", n_points)
    print("Dimensão:", dim)
    print("Clusters:", k)
    print("Partições:", partitions)
    print("Iterações:", iterations)

    base = n_points // partitions
    rem = n_points % partitions

    chunks = []
    for p in range(partitions):
        size = base + (1 if p < rem else 0)
        chunks.append(km_generate_points(size, dim, seed + p, k))

    # Inicializa centroides usando os primeiros pontos da primeira partição.
    first_chunk = compss_wait_on(chunks[0])
    centroids = first_chunk[:k].copy()
    chunks[0] = first_chunk

    for it in range(iterations):
        partials = []
        for chunk in chunks:
            partials.append(km_partial_assignment(chunk, centroids))

        stats = tree_reduce(partials, km_reduce_stats)
        centroids = km_update_centroids(stats, centroids)

    # Cálculo final da inércia usando os centroides finais.
    final_partials = []
    for chunk in chunks:
        final_partials.append(km_partial_assignment(chunk, centroids))

    final_stats = tree_reduce(final_partials, km_reduce_stats)

    centroids = compss_wait_on(centroids)
    sums, counts, inertia = compss_wait_on(final_stats)

    elapsed = time.time() - start

    print("\nResultado KMeans")
    print("Centroides:")
    print(centroids)
    print("Contagem por cluster:")
    print(counts)
    print("Inércia:", inertia)
    print("Tempo total: %.4f s" % elapsed)



def parse_args():
    parser = argparse.ArgumentParser(
        description="Workflows PyCOMPSs: KMeans, Matmul e Lysozyme/GROMACS"
    )

    subparsers = parser.add_subparsers(dest="workflow", required=True)

    # ----------------------------
    # KMeans
    # ----------------------------
    p_km = subparsers.add_parser("kmeans", help="Executa workflow KMeans")
    p_km.add_argument("--points", type=int, default=100000)
    p_km.add_argument("--dim", type=int, default=2)
    p_km.add_argument("--clusters", type=int, default=4)
    p_km.add_argument("--partitions", type=int, default=8)
    p_km.add_argument("--iterations", type=int, default=10)
    p_km.add_argument("--seed", type=int, default=42)

    # ----------------------------
    # Matmul
    # ----------------------------
    p_mm = subparsers.add_parser("matmul", help="Executa workflow Matmul")
    p_mm.add_argument("--size", type=int, default=1024)
    p_mm.add_argument("--block-size", type=int, default=256)
    p_mm.add_argument("--seed", type=int, default=123)

    # ----------------------------
    # Lysozyme/GROMACS
    # ----------------------------
    p_ly = subparsers.add_parser("lysozyme", help="Executa workflow Lysozyme com GROMACS")
    p_ly.add_argument("--pdb", type=str, required=True, help="Arquivo PDB da lisozima, por exemplo 1AKI.pdb")
    p_ly.add_argument("--run-dir", type=str, default="lysozyme_run")

    p_ly.add_argument("--gmx", type=str, default="gmx")
    p_ly.add_argument("--gmx-mpi", type=str, default="gmx_mpi")

    p_ly.add_argument("--forcefield", type=str, default="oplsaa")
    p_ly.add_argument("--water", type=str, default="spce")

    p_ly.add_argument("--box-distance", type=float, default=1.0)
    p_ly.add_argument("--box-type", type=str, default="cubic")

    p_ly.add_argument("--pname", type=str, default="NA")
    p_ly.add_argument("--nname", type=str, default="CL")
    p_ly.add_argument("--ion-group", type=str, default="SOL")

    p_ly.add_argument("--maxwarn", type=int, default=1)

    p_ly.add_argument("--mpi-ranks", type=int, default=1)
    p_ly.add_argument("--ntomp", type=int, default=0)

    return parser.parse_args()



def main():
    args = parse_args()

    if args.workflow == "kmeans":
        run_kmeans(args)
    elif args.workflow == "matmul":
        run_matmul(args)
    elif args.workflow == "lysozyme":
        run_lysozyme(args)
    else:
        raise ValueError("Workflow inválido: %s" % args.workflow)



if __name__ == "__main__":
    main()
