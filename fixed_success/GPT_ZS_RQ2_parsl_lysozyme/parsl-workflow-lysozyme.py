import os



import json



import argparse



from pathlib import Path



import parsl



from parsl import python_app, bash_app



from parsl.config import Config



from parsl.executors import ThreadPoolExecutor



from parsl.data_provider.files import File



@bash_app
def lysozyme_gromacs_task(
    pdb_file: str,
    work_dir: str,
    gmx: str,
    gmx_mpi: str,
    mpi_launcher: str,
    mpi_np: int,
    forcefield: str,
    water: str,
    stdout=None,
    stderr=None,
    outputs=(),
):
    """
    Workflow GROMACS para Lysozyme.

    Etapas:
    1. pdb2gmx: gera topologia.
    2. editconf: define caixa.
    3. solvate: adiciona solvente.
    4. grompp/genion: adiciona íons.
    5. grompp/mdrun: minimização de energia.

    Usa gmx para pré-processamento e tenta usar gmx_mpi para mdrun.
    """
    pdb_abs = os.path.abspath(pdb_file)
    work_abs = os.path.abspath(work_dir)

    return f"""
set -euo pipefail

mkdir -p "{work_abs}"
cd "{work_abs}"

cp "{pdb_abs}" input.pdb

cat > ions.mdp << 'EOF'
integrator  = steep
emtol       = 1000.0
emstep      = 0.01
nsteps      = 500
cutoff-scheme = Verlet
coulombtype = PME
rcoulomb    = 1.0
rvdw        = 1.0
pbc         = xyz
EOF

cat > minim.mdp << 'EOF'
integrator  = steep
emtol       = 1000.0
emstep      = 0.01
nsteps      = 50000
cutoff-scheme = Verlet
coulombtype = PME
rcoulomb    = 1.0
rvdw        = 1.0
pbc         = xyz
EOF

echo "[LYSOZYME] Gerando topologia com pdb2gmx"

"{gmx}" pdb2gmx \\
    -f input.pdb \\
    -o processed.gro \\
    -p topol.top \\
    -i posre.itp \\
    -ff "{forcefield}" \\
    -water "{water}" \\
    -ignh

echo "[LYSOZYME] Criando caixa"

"{gmx}" editconf \\
    -f processed.gro \\
    -o boxed.gro \\
    -c \\
    -d 1.0 \\
    -bt cubic

echo "[LYSOZYME] Solvatando sistema"

"{gmx}" solvate \\
    -cp boxed.gro \\
    -cs spc216.gro \\
    -o solvated.gro \\
    -p topol.top

echo "[LYSOZYME] Preparando adição de íons"

"{gmx}" grompp \\
    -f ions.mdp \\
    -c solvated.gro \\
    -p topol.top \\
    -o ions.tpr \\
    -maxwarn 2

echo "[LYSOZYME] Adicionando íons"

printf "SOL\\n" | "{gmx}" genion \\
    -s ions.tpr \\
    -o solv_ions.gro \\
    -p topol.top \\
    -pname NA \\
    -nname CL \\
    -neutral

echo "[LYSOZYME] Preparando minimização de energia"

"{gmx}" grompp \\
    -f minim.mdp \\
    -c solv_ions.gro \\
    -p topol.top \\
    -o em.tpr \\
    -maxwarn 2

echo "[LYSOZYME] Executando mdrun"

if command -v "{gmx_mpi}" >/dev/null 2>&1; then
    "{mpi_launcher}" -np {mpi_np} "{gmx_mpi}" mdrun -deffnm em
else
    echo "[LYSOZYME] gmx_mpi não encontrado. Usando gmx mdrun."
    "{gmx}" mdrun -deffnm em
fi

echo "[LYSOZYME] Extraindo energia potencial"

printf "Potential\\n0\\n" | "{gmx}" energy \\
    -f em.edr \\
    -o potential.xvg || true

echo "[LYSOZYME] Workflow finalizado"
echo "[LYSOZYME] Arquivos principais:"
ls -lh em.gro em.edr em.log em.tpr topol.top || true
"""



def run_lysozyme_workflow(args):
    """
    Workflow independente para Lysozyme usando GROMACS.
    """
    out_dir = Path(args.outdir) / "lysozyme"
    out_dir.mkdir(parents=True, exist_ok=True)

    pdb_file = Path(args.lysozyme_pdb)

    if not pdb_file.exists():
        raise FileNotFoundError(
            f"Arquivo PDB não encontrado: {pdb_file}\n"
            f"Informe um PDB válido com --lysozyme-pdb. "
            f"Exemplo: 1AKI.pdb para o tutorial clássico de lysozyme."
        )

    stdout_file = out_dir / "lysozyme.stdout"
    stderr_file = out_dir / "lysozyme.stderr"

    fut = lysozyme_gromacs_task(
        pdb_file=str(pdb_file),
        work_dir=str(out_dir),
        gmx=args.gmx,
        gmx_mpi=args.gmx_mpi,
        mpi_launcher=args.mpi_launcher,
        mpi_np=args.mpi_np,
        forcefield=args.forcefield,
        water=args.water,
        stdout=str(stdout_file),
        stderr=str(stderr_file),
        outputs=[File(str(out_dir / "em.gro"))],
    )

    fut.result()

    print(f"[LYSOZYME] Workflow finalizado.")
    print(f"[LYSOZYME] Diretório: {out_dir}")
    print(f"[LYSOZYME] stdout: {stdout_file}")
    print(f"[LYSOZYME] stderr: {stderr_file}")



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
