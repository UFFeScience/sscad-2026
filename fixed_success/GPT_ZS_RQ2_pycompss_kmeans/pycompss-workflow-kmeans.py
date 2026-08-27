import os



import shlex



import argparse



import subprocess



from pycompss.api.task import task



from pycompss.api.parameter import FILE_IN, DIRECTORY_OUT, DIRECTORY_INOUT



from pycompss.api.api import compss_wait_on, compss_barrier



@task(returns=1)
def generate_points(n_points, n_dims, seed):
    import numpy as np

    rng = np.random.default_rng(seed)
    points = rng.normal(size=(n_points, n_dims)).astype("float64")
    return points



@task(returns=1)
def kmeans_partial(points, centroids):
    import numpy as np

    k = centroids.shape[0]
    distances = ((points[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    labels = np.argmin(distances, axis=1)

    sums = np.zeros_like(centroids)
    counts = np.zeros(k, dtype=np.int64)

    np.add.at(sums, labels, points)
    counts += np.bincount(labels, minlength=k)

    return sums, counts



def run_kmeans(args):
    import numpy as np

    n_points = args.points
    n_dims = args.dims
    k = args.clusters
    n_partitions = args.partitions
    iterations = args.iterations

    print("Executando workflow KMeans com PyCOMPSs")

    sizes = [n_points // n_partitions] * n_partitions
    for i in range(n_points % n_partitions):
        sizes[i] += 1

    partitions = [
        generate_points(sizes[i], n_dims, args.seed + i)
        for i in range(n_partitions)
    ]

    # Inicialização simples dos centróides usando a primeira partição
    # (gera um future dedicado em vez de reaproveitar partitions[0], que
    # seria referenciado de novo como argumento de task no loop abaixo)
    first_partition = compss_wait_on(generate_points(sizes[0], n_dims, args.seed))
    if k > first_partition.shape[0]:
        raise ValueError("Número de clusters maior que o tamanho da primeira partição.")

    centroids = first_partition[:k].copy()

    for it in range(iterations):
        partials = [
            kmeans_partial(partition, centroids)
            for partition in partitions
        ]

        partials = compss_wait_on(partials)

        total_sums = np.zeros_like(centroids)
        total_counts = np.zeros(k, dtype=np.int64)

        for sums, counts in partials:
            total_sums += sums
            total_counts += counts

        new_centroids = centroids.copy()
        non_empty = total_counts > 0
        new_centroids[non_empty] = (
            total_sums[non_empty] / total_counts[non_empty, None]
        )

        shift = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids

        print(f"Iteração {it + 1}/{iterations} - deslocamento = {shift:.6f}")

    print("Centróides finais:")
    print(centroids)



def build_parser():
    parser = argparse.ArgumentParser(
        description="Workflows PyCOMPSs: KMeans, Matmul e Lysozyme/GROMACS"
    )

    subparsers = parser.add_subparsers(dest="workflow", required=True)

    # KMeans
    p_kmeans = subparsers.add_parser("kmeans")
    p_kmeans.add_argument("--points", type=int, default=100000)
    p_kmeans.add_argument("--dims", type=int, default=3)
    p_kmeans.add_argument("--clusters", type=int, default=4)
    p_kmeans.add_argument("--partitions", type=int, default=8)
    p_kmeans.add_argument("--iterations", type=int, default=10)
    p_kmeans.add_argument("--seed", type=int, default=42)

    # Matmul
    p_matmul = subparsers.add_parser("matmul")
    p_matmul.add_argument("--blocks", type=int, default=4)
    p_matmul.add_argument("--block-size", type=int, default=512)
    p_matmul.add_argument("--seed", type=int, default=42)

    # Lysozyme
    p_lyso = subparsers.add_parser("lysozyme")
    p_lyso.add_argument("--pdb", required=True, help="Arquivo PDB da lisozima, por exemplo 1aki.pdb")
    p_lyso.add_argument("--workdir", default="lysozyme_work")

    p_lyso.add_argument("--gmx", default="gmx", help="Binário gmx do GROMACS")
    p_lyso.add_argument("--gmx-mpi", default="gmx_mpi", help="Binário gmx_mpi do GROMACS")

    p_lyso.add_argument("--forcefield", default="oplsaa")
    p_lyso.add_argument("--water", default="spce")

    p_lyso.add_argument("--box-distance", type=float, default=1.0)
    p_lyso.add_argument("--box-type", default="cubic")

    p_lyso.add_argument("--solvent-group", default="SOL")
    p_lyso.add_argument("--maxwarn", type=int, default=1)

    p_lyso.add_argument(
        "--mpi-launcher",
        default="",
        help='Opcional. Exemplo: "mpirun -np 4" ou "srun -n 4"'
    )
    p_lyso.add_argument("--ntomp", type=int, default=1)

    return parser



def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.workflow == "kmeans":
        run_kmeans(args)

    elif args.workflow == "matmul":
        run_matmul(args)

    elif args.workflow == "lysozyme":
        run_lysozyme(args)

    compss_barrier()



if __name__ == "__main__":
    main()
