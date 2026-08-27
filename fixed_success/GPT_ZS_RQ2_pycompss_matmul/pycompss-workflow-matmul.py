import os



import argparse



import subprocess



import shutil



import numpy as np



from pycompss.api.task import task



from pycompss.api.parameter import FILE_IN, FILE_OUT



from pycompss.api.api import compss_wait_on, compss_barrier



@task(returns=1)
def generate_block(block_size, seed):
    rng = np.random.default_rng(seed)
    return rng.random((block_size, block_size), dtype=np.float64)



@task(returns=1)
def matmul_block(a, b):
    return np.matmul(a, b)



@task(returns=1)
def add_blocks(blocks):
    result = np.zeros_like(blocks[0])
    for b in blocks:
        result += b
    return result



def workflow_matmul(args):
    print("=== Workflow Matmul ===")

    n_blocks = args.matrix_size // args.block_size

    if args.matrix_size % args.block_size != 0:
        raise ValueError("matrix-size deve ser múltiplo de block-size")

    A = []
    B = []

    for i in range(n_blocks):
        row_a = []
        row_b = []
        for j in range(n_blocks):
            row_a.append(generate_block(args.block_size, args.seed + i * n_blocks + j))
            row_b.append(generate_block(args.block_size, args.seed + 10000 + i * n_blocks + j))
        A.append(row_a)
        B.append(row_b)

    C = []

    for i in range(n_blocks):
        row_c = []
        for j in range(n_blocks):
            partials = []
            for k in range(n_blocks):
                partials.append(matmul_block(A[i][k], B[k][j]))
            # add_blocks recebe "partials" como lista de futures aninhados;
            # o PyCOMPSs nao resolve futures aninhados dentro de uma lista
            # Python automaticamente, entao sincronizamos antes de passar.
            partials = compss_wait_on(partials)
            row_c.append(add_blocks(partials))
        C.append(row_c)

    C = compss_wait_on(C)

    print("Multiplicação concluída.")
    print(f"Matriz C possui {n_blocks} x {n_blocks} blocos.")
    print("Bloco C[0][0]:")
    print(C[0][0])



def parse_args():
    parser = argparse.ArgumentParser(
        description="Workflows PyCOMPSs: KMeans, Matmul e Lysozyme/GROMACS"
    )

    subparsers = parser.add_subparsers(
        dest="workflow",
        required=True
    )

    # -------------------------
    # KMeans
    # -------------------------
    p_kmeans = subparsers.add_parser("kmeans")

    p_kmeans.add_argument("--points", type=int, default=100000)
    p_kmeans.add_argument("--dims", type=int, default=2)
    p_kmeans.add_argument("--k", type=int, default=4)
    p_kmeans.add_argument("--chunks", type=int, default=8)
    p_kmeans.add_argument("--iterations", type=int, default=10)
    p_kmeans.add_argument("--seed", type=int, default=1234)

    # -------------------------
    # Matmul
    # -------------------------
    p_matmul = subparsers.add_parser("matmul")

    p_matmul.add_argument("--matrix-size", type=int, default=1024)
    p_matmul.add_argument("--block-size", type=int, default=256)
    p_matmul.add_argument("--seed", type=int, default=1234)

    # -------------------------
    # Lysozyme
    # -------------------------
    p_lyso = subparsers.add_parser("lysozyme")

    p_lyso.add_argument(
        "--pdb",
        required=True,
        help="Arquivo PDB de entrada da lisozima, por exemplo 1AKI_clean.pdb"
    )

    p_lyso.add_argument(
        "--workdir",
        default="lysozyme_workdir",
        help="Diretório de trabalho"
    )

    p_lyso.add_argument(
        "--gmx",
        default="gmx",
        help="Executável serial do GROMACS"
    )

    p_lyso.add_argument(
        "--gmx-mpi",
        default="gmx_mpi",
        help="Executável MPI do GROMACS"
    )

    p_lyso.add_argument(
        "--mpi-ranks",
        type=int,
        default=4,
        help="Número de ranks MPI para o mdrun"
    )

    p_lyso.add_argument(
        "--forcefield",
        default="amber99sb-ildn",
        help="Campo de força usado no pdb2gmx"
    )

    p_lyso.add_argument(
        "--water",
        default="spce",
        help="Modelo de água usado no pdb2gmx"
    )

    return parser.parse_args()



def main():
    args = parse_args()

    if args.workflow == "kmeans":
        workflow_kmeans(args)

    elif args.workflow == "matmul":
        workflow_matmul(args)

    elif args.workflow == "lysozyme":
        workflow_lysozyme(args)

    else:
        raise ValueError(f"Workflow desconhecido: {args.workflow}")



if __name__ == "__main__":
    main()
