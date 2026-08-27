import os



import json



import argparse



import shlex



from pathlib import Path



import parsl



from parsl import python_app, bash_app



from parsl.config import Config



from parsl.executors.threads import ThreadPoolExecutor



from parsl.data_provider.files import File



def load_parsl(max_threads: int):
    try:
        parsl.clear()
    except Exception:
        pass

    config = Config(
        executors=[
            ThreadPoolExecutor(
                label="local_threads",
                max_threads=max_threads,
            )
        ]
    )

    parsl.load(config)



@python_app
def kmeans_trial(
    n_samples: int,
    n_features: int,
    n_clusters: int,
    seed_dataset: int,
    seed_kmeans: int,
    max_iter: int,
    outdir: str,
):
    import os
    import json
    import numpy as np
    from sklearn.datasets import make_blobs
    from sklearn.cluster import KMeans

    os.makedirs(outdir, exist_ok=True)

    X, _ = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        random_state=seed_dataset,
    )

    model = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=1,
        max_iter=max_iter,
        random_state=seed_kmeans,
    )

    labels = model.fit_predict(X)

    output_npz = os.path.join(outdir, f"kmeans_seed_{seed_kmeans}.npz")
    np.savez_compressed(
        output_npz,
        centers=model.cluster_centers_,
        labels=labels,
        inertia=model.inertia_,
    )

    return {
        "seed_kmeans": seed_kmeans,
        "inertia": float(model.inertia_),
        "iterations": int(model.n_iter_),
        "output": output_npz,
    }



@python_app
def select_best_kmeans(results, summary_file: str):
    import os
    import json

    os.makedirs(os.path.dirname(summary_file), exist_ok=True)

    results = [r.result() if hasattr(r, "result") else r for r in results]

    best = min(results, key=lambda x: x["inertia"])

    summary = {
        "best": best,
        "all_trials": results,
    }

    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    return summary



def run_kmeans_workflow(args):
    outdir = os.path.abspath(os.path.join(args.outdir, "kmeans"))
    os.makedirs(outdir, exist_ok=True)

    futures = []

    for i in range(args.trials):
        seed_kmeans = args.seed + i

        fut = kmeans_trial(
            args.n_samples,
            args.n_features,
            args.n_clusters,
            args.seed,
            seed_kmeans,
            args.max_iter,
            outdir,
        )
        futures.append(fut)

    summary_file = os.path.join(outdir, "summary.json")
    final = select_best_kmeans(futures, summary_file)

    result = final.result()

    print(json.dumps(result, indent=2))
    print(f"\nResumo salvo em: {summary_file}")



def build_parser():
    parser = argparse.ArgumentParser(
        description="Workflows Parsl para KMeans, Matmul e Lysozyme/GROMACS."
    )

    parser.add_argument(
        "--outdir",
        default="results",
        help="Diretório base de saída.",
    )

    parser.add_argument(
        "--max-threads",
        type=int,
        default=os.cpu_count() or 4,
        help="Número máximo de threads usadas pelo executor Parsl local.",
    )

    sub = parser.add_subparsers(dest="workflow", required=True)

    # KMeans
    p_kmeans = sub.add_parser("kmeans", help="Executa workflow KMeans.")
    p_kmeans.add_argument("--n-samples", type=int, default=100000)
    p_kmeans.add_argument("--n-features", type=int, default=16)
    p_kmeans.add_argument("--n-clusters", type=int, default=8)
    p_kmeans.add_argument("--trials", type=int, default=8)
    p_kmeans.add_argument("--max-iter", type=int, default=300)
    p_kmeans.add_argument("--seed", type=int, default=42)

    # Matmul
    p_matmul = sub.add_parser("matmul", help="Executa workflow Matmul por blocos.")
    p_matmul.add_argument("--n", type=int, default=1024)
    p_matmul.add_argument("--block-size", type=int, default=256)
    p_matmul.add_argument("--seed", type=int, default=42)

    # Lysozyme
    p_lyso = sub.add_parser("lysozyme", help="Executa workflow Lysozyme com GROMACS.")
    p_lyso.add_argument(
        "--pdb",
        default=None,
        help="Arquivo PDB de entrada. Se omitido, baixa 1AKI do RCSB.",
    )
    p_lyso.add_argument(
        "--forcefield",
        default="amber99sb-ildn",
        help="Campo de força usado no gmx pdb2gmx.",
    )
    p_lyso.add_argument(
        "--water",
        default="tip3p",
        help="Modelo de água usado no gmx pdb2gmx.",
    )
    p_lyso.add_argument(
        "--box-distance",
        type=float,
        default=1.0,
        help="Distância da proteína até a borda da caixa em nm.",
    )
    p_lyso.add_argument(
        "--mpi-prefix",
        default="",
        help='Prefixo MPI para gmx_mpi. Exemplo: "mpirun -np 4".',
    )
    p_lyso.add_argument(
        "--mdrun-extra",
        default="",
        help='Argumentos extras para gmx_mpi mdrun. Exemplo: "-ntomp 2".',
    )
    p_lyso.add_argument("--nsteps-min", type=int, default=500)
    p_lyso.add_argument("--nsteps-eq", type=int, default=500)
    p_lyso.add_argument("--nsteps-md", type=int, default=1000)
    p_lyso.add_argument("--maxwarn", type=int, default=1)

    return parser



def main():
    parser = build_parser()
    args = parser.parse_args()

    load_parsl(args.max_threads)

    try:
        if args.workflow == "kmeans":
            run_kmeans_workflow(args)
        elif args.workflow == "matmul":
            run_matmul_workflow(args)
        elif args.workflow == "lysozyme":
            run_lysozyme_workflow(args)
        else:
            raise ValueError(f"Workflow inválido: {args.workflow}")
    finally:
        try:
            parsl.dfk().cleanup()
        except Exception:
            pass
        try:
            parsl.clear()
        except Exception:
            pass



if __name__ == "__main__":
    main()
