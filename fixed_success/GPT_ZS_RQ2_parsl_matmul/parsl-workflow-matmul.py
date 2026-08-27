import os



import json



import argparse



from pathlib import Path



import parsl



from parsl import python_app, bash_app



from parsl.config import Config



from parsl.executors import ThreadPoolExecutor



from parsl.data_provider.files import File



@python_app
def matmul_block_task(
    row_start: int,
    row_end: int,
    n_cols_a: int,
    n_cols_b: int,
    seed: int,
    b_file: str,
    block_file: str,
):
    """
    Multiplica um bloco de linhas de A por B.

    A é gerada por bloco, de forma independente.
    B é carregada de um arquivo .npy compartilhado.
    """
    import numpy as np
    from pathlib import Path

    Path(block_file).parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed + row_start)
    a_block = rng.standard_normal((row_end - row_start, n_cols_a))

    B = np.load(b_file)

    c_block = a_block @ B

    np.save(block_file, c_block)

    return {
        "algorithm": "matmul",
        "row_start": row_start,
        "row_end": row_end,
        "block_file": block_file,
        "shape": c_block.shape,
        "checksum": float(c_block.sum()),
    }



def run_matmul_workflow(args):
    """
    Workflow independente para multiplicação de matrizes.

    Calcula C = A x B em blocos de linhas usando várias tarefas Parsl.
    """
    import numpy as np

    out_dir = Path(args.outdir) / "matmul"
    blocks_dir = out_dir / "blocks"
    out_dir.mkdir(parents=True, exist_ok=True)
    blocks_dir.mkdir(parents=True, exist_ok=True)

    n = args.matmul_n
    m = args.matmul_m
    p = args.matmul_p
    block_size = args.matmul_block_size

    rng = np.random.default_rng(args.matmul_seed)
    B = rng.standard_normal((m, p))

    b_file = out_dir / "B.npy"
    np.save(b_file, B)

    futures = []

    for row_start in range(0, n, block_size):
        row_end = min(row_start + block_size, n)
        block_file = blocks_dir / f"C_{row_start}_{row_end}.npy"

        fut = matmul_block_task(
            row_start=row_start,
            row_end=row_end,
            n_cols_a=m,
            n_cols_b=p,
            seed=args.matmul_seed,
            b_file=str(b_file),
            block_file=str(block_file),
        )
        futures.append(fut)

    metadata = [f.result() for f in futures]

    c_blocks = [np.load(item["block_file"]) for item in sorted(metadata, key=lambda x: x["row_start"])]
    C = np.vstack(c_blocks)

    c_file = out_dir / "C.npy"
    np.save(c_file, C)

    result_file = out_dir / "matmul_result.json"

    result = {
        "algorithm": "matmul",
        "A_shape": [n, m],
        "B_shape": [m, p],
        "C_shape": list(C.shape),
        "block_size": block_size,
        "num_blocks": len(metadata),
        "checksum": float(C.sum()),
        "C_file": str(c_file),
        "blocks": metadata,
    }

    with open(result_file, "w") as f:
        json.dump(result, f, indent=2)

    print(f"[MATMUL] Resultado salvo em: {result_file}")
    print(f"[MATMUL] Matriz C salva em: {c_file}")
    print(f"[MATMUL] Checksum: {result['checksum']}")



def parse_args():
    parser = argparse.ArgumentParser(
        description="Workflows Parsl para KMeans, Matmul e Lysozyme/GROMACS."
    )

    parser.add_argument(
        "--workflow",
        choices=["kmeans", "matmul", "lysozyme", "all"],
        required=True,
        help="Workflow a executar.",
    )

    parser.add_argument(
        "--outdir",
        default="outputs",
        help="Diretório de saída.",
    )

    parser.add_argument(
        "--parsl-run-dir",
        default="parsl_runinfo",
        help="Diretório de runinfo do Parsl.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Número de workers locais do Parsl.",
    )

    # KMeans
    parser.add_argument("--kmeans-samples", type=int, default=5000)
    parser.add_argument("--kmeans-features", type=int, default=8)
    parser.add_argument("--kmeans-clusters", type=int, default=4)
    parser.add_argument("--kmeans-max-iter", type=int, default=100)
    parser.add_argument("--kmeans-tol", type=float, default=1e-4)
    parser.add_argument("--kmeans-runs", type=int, default=4)
    parser.add_argument("--kmeans-data-seed", type=int, default=123)
    parser.add_argument("--kmeans-init-seed", type=int, default=1000)

    # Matmul
    parser.add_argument("--matmul-n", type=int, default=2048)
    parser.add_argument("--matmul-m", type=int, default=2048)
    parser.add_argument("--matmul-p", type=int, default=2048)
    parser.add_argument("--matmul-block-size", type=int, default=256)
    parser.add_argument("--matmul-seed", type=int, default=42)

    # Lysozyme/GROMACS
    parser.add_argument(
        "--lysozyme-pdb",
        default="1AKI.pdb",
        help="Arquivo PDB de entrada, por exemplo 1AKI.pdb.",
    )

    parser.add_argument(
        "--gmx",
        default="gmx",
        help="Executável gmx do GROMACS.",
    )

    parser.add_argument(
        "--gmx-mpi",
        default="gmx_mpi",
        help="Executável gmx_mpi do GROMACS.",
    )

    parser.add_argument(
        "--mpi-launcher",
        default="mpirun",
        help="Launcher MPI, por exemplo mpirun ou srun.",
    )

    parser.add_argument(
        "--mpi-np",
        type=int,
        default=2,
        help="Número de processos MPI para gmx_mpi mdrun.",
    )

    parser.add_argument(
        "--forcefield",
        default="oplsaa",
        help="Campo de força usado no pdb2gmx. Exemplo: oplsaa.",
    )

    parser.add_argument(
        "--water",
        default="spce",
        help="Modelo de água usado no pdb2gmx. Exemplo: spce.",
    )

    return parser.parse_args()



def load_parsl(workers=4, run_dir="runinfo"):
    config = Config(
        executors=[ThreadPoolExecutor(max_threads=workers, label="local")],
        run_dir=run_dir,
    )
    parsl.load(config)


def main():
    args = parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    Path(args.parsl_run_dir).mkdir(parents=True, exist_ok=True)

    load_parsl(workers=args.workers, run_dir=args.parsl_run_dir)

    try:
        if args.workflow == "kmeans":
            run_kmeans_workflow(args)

        elif args.workflow == "matmul":
            run_matmul_workflow(args)

        elif args.workflow == "lysozyme":
            run_lysozyme_workflow(args)

        elif args.workflow == "all":
            run_kmeans_workflow(args)
            run_matmul_workflow(args)
            run_lysozyme_workflow(args)

    finally:
        parsl.dfk().cleanup()



if __name__ == "__main__":
    main()
