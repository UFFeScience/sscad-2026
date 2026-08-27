import argparse



from pathlib import Path



import parsl



from parsl.app.app import python_app, bash_app



from parsl.config import Config



from parsl.executors.threads import ThreadPoolExecutor



from parsl.data_provider.files import File



@bash_app
def pdb2gmx_app(outdir, gmx, forcefield, water, inputs=[], outputs=[]):
    import shlex

    q = lambda x: shlex.quote(str(x))

    pdb = q(inputs[0])
    od = q(outdir)
    gmx_cmd = q(gmx)
    ff = q(forcefield)
    wat = q(water)

    return f"""
set -euo pipefail

mkdir -p {od}
cp {pdb} {od}/input.pdb

cd {od}

{gmx_cmd} pdb2gmx \\
    -f input.pdb \\
    -o processed.gro \\
    -p topol_pdb2gmx.top \\
    -i posre.itp \\
    -ff {ff} \\
    -water {wat} \\
    -ignh

test -s processed.gro
test -s topol_pdb2gmx.top
test -s posre.itp
"""



@bash_app
def editconf_app(outdir, gmx, box_distance, box_type, inputs=[], outputs=[]):
    import shlex

    q = lambda x: shlex.quote(str(x))

    od = q(outdir)
    gmx_cmd = q(gmx)
    dist = q(box_distance)
    btype = q(box_type)

    return f"""
set -euo pipefail

cd {od}

{gmx_cmd} editconf \\
    -f processed.gro \\
    -o boxed.gro \\
    -c \\
    -d {dist} \\
    -bt {btype}

test -s boxed.gro
"""



@bash_app
def solvate_app(outdir, gmx, inputs=[], outputs=[]):
    import shlex

    q = lambda x: shlex.quote(str(x))

    od = q(outdir)
    gmx_cmd = q(gmx)

    return f"""
set -euo pipefail

cd {od}

cp topol_pdb2gmx.top topol_solv.top

{gmx_cmd} solvate \\
    -cp boxed.gro \\
    -cs spc216.gro \\
    -o solv.gro \\
    -p topol_solv.top

test -s solv.gro
test -s topol_solv.top
"""



@bash_app
def grompp_app(outdir, gmx, output_tpr, maxwarn, inputs=[], outputs=[]):
    import os
    import shlex

    q = lambda x: shlex.quote(str(x))

    od = q(outdir)
    gmx_cmd = q(gmx)

    mdp = q(inputs[0])
    coord = q(os.path.basename(str(inputs[1])))
    topol = q(os.path.basename(str(inputs[2])))
    tpr = q(output_tpr)

    maxwarn_opt = ""
    if int(maxwarn) > 0:
        maxwarn_opt = f"-maxwarn {int(maxwarn)}"

    return f"""
set -euo pipefail

cd {od}

{gmx_cmd} grompp \\
    -f {mdp} \\
    -c {coord} \\
    -p {topol} \\
    -o {tpr} \\
    {maxwarn_opt}

test -s {tpr}
"""



@bash_app
def genion_app(outdir, gmx, group_name, inputs=[], outputs=[]):
    import shlex

    q = lambda x: shlex.quote(str(x))

    od = q(outdir)
    gmx_cmd = q(gmx)
    group = q(group_name)

    return f"""
set -euo pipefail

cd {od}

cp topol_solv.top topol_ions.top

printf "%s\\n" {group} | {gmx_cmd} genion \\
    -s ions.tpr \\
    -o solv_ions.gro \\
    -p topol_ions.top \\
    -pname NA \\
    -nname CL \\
    -neutral

test -s solv_ions.gro
test -s topol_ions.top
"""



@bash_app
def mdrun_app(outdir, gmx_mpi, mpi_prefix, inputs=[], outputs=[]):
    import shlex

    q = lambda x: shlex.quote(str(x))

    od = q(outdir)
    gmx_mpi_cmd = q(gmx_mpi)

    # mpi_prefix pode ser, por exemplo:
    # "mpirun -np 4", "srun -n 4" ou "".
    prefix = str(mpi_prefix).strip()

    if prefix:
        run_cmd = f"{prefix} {gmx_mpi_cmd} mdrun"
    else:
        run_cmd = f"{gmx_mpi_cmd} mdrun"

    return f"""
set -euo pipefail

cd {od}

{run_cmd} \\
    -s em.tpr \\
    -deffnm em

test -s em.gro
test -s em.log
"""



def write_lysozyme_mdp_files(outdir: Path, em_nsteps: int):
    outdir.mkdir(parents=True, exist_ok=True)

    ions_mdp = outdir / "ions.mdp"
    minim_mdp = outdir / "minim.mdp"

    ions_mdp.write_text(
        """; ions.mdp
integrator      = steep
emtol           = 1000.0
emstep          = 0.01
nsteps          = 500
cutoff-scheme   = Verlet
coulombtype     = PME
rcoulomb        = 1.0
rvdw            = 1.0
pbc             = xyz
"""
    )

    minim_mdp.write_text(
        f"""; minim.mdp
integrator      = steep
emtol           = 1000.0
emstep          = 0.01
nsteps          = {em_nsteps}
cutoff-scheme   = Verlet
coulombtype     = PME
rcoulomb        = 1.0
rvdw            = 1.0
pbc             = xyz
"""
    )

    return ions_mdp.resolve(), minim_mdp.resolve()



def run_lysozyme(args):
    if args.pdb is None:
        raise ValueError("Para o workflow lysozyme, informe --pdb arquivo.pdb")

    outdir = Path(args.lysozyme_out).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    pdb = Path(args.pdb).resolve()
    if not pdb.exists():
        raise FileNotFoundError(f"PDB não encontrado: {pdb}")

    ions_mdp, minim_mdp = write_lysozyme_mdp_files(
        outdir,
        em_nsteps=args.em_nsteps,
    )

    pdb2gmx = pdb2gmx_app(
        str(outdir),
        args.gmx,
        args.forcefield,
        args.water,
        inputs=[
            File(str(pdb)),
        ],
        outputs=[
            File(str(outdir / "processed.gro")),
            File(str(outdir / "topol_pdb2gmx.top")),
            File(str(outdir / "posre.itp")),
        ],
    )

    editconf = editconf_app(
        str(outdir),
        args.gmx,
        args.box_distance,
        args.box_type,
        inputs=[
            pdb2gmx.outputs[0],
        ],
        outputs=[
            File(str(outdir / "boxed.gro")),
        ],
    )

    solvate = solvate_app(
        str(outdir),
        args.gmx,
        inputs=[
            editconf.outputs[0],
            pdb2gmx.outputs[1],
            pdb2gmx.outputs[2],
        ],
        outputs=[
            File(str(outdir / "solv.gro")),
            File(str(outdir / "topol_solv.top")),
        ],
    )

    ions_tpr = grompp_app(
        str(outdir),
        args.gmx,
        "ions.tpr",
        args.maxwarn,
        inputs=[
            File(str(ions_mdp)),
            solvate.outputs[0],
            solvate.outputs[1],
            pdb2gmx.outputs[2],
        ],
        outputs=[
            File(str(outdir / "ions.tpr")),
        ],
    )

    genion = genion_app(
        str(outdir),
        args.gmx,
        args.genion_group,
        inputs=[
            ions_tpr.outputs[0],
            solvate.outputs[1],
        ],
        outputs=[
            File(str(outdir / "solv_ions.gro")),
            File(str(outdir / "topol_ions.top")),
        ],
    )

    em_tpr = grompp_app(
        str(outdir),
        args.gmx,
        "em.tpr",
        args.maxwarn,
        inputs=[
            File(str(minim_mdp)),
            genion.outputs[0],
            genion.outputs[1],
            pdb2gmx.outputs[2],
        ],
        outputs=[
            File(str(outdir / "em.tpr")),
        ],
    )

    em = mdrun_app(
        str(outdir),
        args.gmx_mpi,
        args.mpi_prefix,
        inputs=[
            em_tpr.outputs[0],
        ],
        outputs=[
            File(str(outdir / "em.gro")),
            File(str(outdir / "em.edr")),
            File(str(outdir / "em.log")),
        ],
    )

    return em



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
        "--workers",
        type=int,
        default=4,
        help="Número de threads do executor local Parsl.",
    )

    parser.add_argument(
        "--run-dir",
        default="parsl_runinfo",
        help="Diretório de logs/metadados do Parsl.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Semente aleatória para KMeans e Matmul.",
    )

    # KMeans
    parser.add_argument("--kmeans-out", default="runs/kmeans")
    parser.add_argument("--kmeans-samples", type=int, default=100000)
    parser.add_argument("--kmeans-features", type=int, default=32)
    parser.add_argument("--kmeans-clusters", type=int, default=8)

    # Matmul
    parser.add_argument("--matmul-out", default="runs/matmul")
    parser.add_argument("--matmul-n", type=int, default=1024)
    parser.add_argument("--matmul-m", type=int, default=1024)
    parser.add_argument("--matmul-p", type=int, default=1024)

    # Lysozyme / GROMACS
    parser.add_argument("--lysozyme-out", default="runs/lysozyme")
    parser.add_argument("--pdb", default=None, help="Arquivo PDB da lisozima, por exemplo 1AKI.pdb.")

    parser.add_argument("--gmx", default="gmx", help="Executável serial do GROMACS.")
    parser.add_argument("--gmx-mpi", default="gmx_mpi", help="Executável MPI do GROMACS.")

    parser.add_argument(
        "--mpi-prefix",
        default="mpirun -np 4",
        help='Prefixo MPI. Exemplos: "mpirun -np 4", "srun -n 4" ou "".',
    )

    parser.add_argument(
        "--forcefield",
        default="oplsaa",
        help="Campo de força usado no pdb2gmx.",
    )

    parser.add_argument(
        "--water",
        default="spce",
        help="Modelo de água usado no pdb2gmx.",
    )

    parser.add_argument("--box-distance", default="1.0")
    parser.add_argument("--box-type", default="cubic")
    parser.add_argument("--em-nsteps", type=int, default=5000)
    parser.add_argument("--maxwarn", type=int, default=1)
    parser.add_argument("--genion-group", default="SOL")

    return parser.parse_args()



def main():
    args = parse_args()

    config = Config(
        executors=[
            ThreadPoolExecutor(
                label="local_threads",
                max_threads=args.workers,
            )
        ],
        run_dir=args.run_dir,
    )

    parsl.load(config)

    futures = []

    try:
        if args.workflow in ["kmeans", "all"]:
            futures.append(("kmeans", run_kmeans(args)))

        if args.workflow in ["matmul", "all"]:
            futures.append(("matmul", run_matmul(args)))

        if args.workflow in ["lysozyme", "all"]:
            futures.append(("lysozyme", run_lysozyme(args)))

        for name, fut in futures:
            print(f"[INFO] Aguardando workflow: {name}")
            result = fut.result()
            print(f"[OK] Workflow finalizado: {name}")
            print(result)

    finally:
        parsl.dfk().cleanup()
        parsl.clear()



if __name__ == "__main__":
    main()
