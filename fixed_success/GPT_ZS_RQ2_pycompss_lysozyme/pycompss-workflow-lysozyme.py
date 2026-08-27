import os



import shutil



import argparse



import subprocess



import numpy as np



from pycompss.api.task import task



from pycompss.api.parameter import FILE_IN, FILE_OUT, FILE_INOUT



from pycompss.api.api import compss_wait_on, compss_barrier



def run_cmd(cmd, cwd=None, stdin_text=None):
    print("[CMD]", " ".join(cmd))
    subprocess.run(
        cmd,
        cwd=cwd,
        input=stdin_text,
        text=True,
        check=True,
    )



def basename(path):
    return os.path.basename(path)



def dirname(path):
    return os.path.dirname(os.path.abspath(path))



def write_mdp_files(workdir):
    ions_mdp = os.path.join(workdir, "ions.mdp")
    minim_mdp = os.path.join(workdir, "minim.mdp")

    with open(ions_mdp, "w") as f:
        f.write(
            """
integrator      = steep
emtol           = 1000.0
emstep          = 0.01
nsteps          = 500
nstlist         = 1
cutoff-scheme   = Verlet
ns_type         = grid
coulombtype     = PME
rcoulomb        = 1.0
rvdw            = 1.0
pbc             = xyz
"""
        )

    with open(minim_mdp, "w") as f:
        f.write(
            """
integrator      = steep
emtol           = 1000.0
emstep          = 0.01
nsteps          = 50000
nstlist         = 10
cutoff-scheme   = Verlet
ns_type         = grid
coulombtype     = PME
rcoulomb        = 1.0
rvdw            = 1.0
pbc             = xyz
"""
        )

    return ions_mdp, minim_mdp



@task(pdb=FILE_IN, gro=FILE_OUT, top=FILE_OUT, posre=FILE_OUT)
def gmx_pdb2gmx_task(gmx, pdb, gro, top, posre, force_field, water_model):
    cwd = dirname(gro)

    cmd = [
        gmx,
        "pdb2gmx",
        "-f", basename(pdb),
        "-o", basename(gro),
        "-p", basename(top),
        "-i", basename(posre),
        "-ff", force_field,
        "-water", water_model,
    ]

    run_cmd(cmd, cwd=cwd)



@task(inp=FILE_IN, out=FILE_OUT)
def gmx_editconf_task(gmx, inp, out, distance, box_type):
    cwd = dirname(out)

    cmd = [
        gmx,
        "editconf",
        "-f", basename(inp),
        "-o", basename(out),
        "-c",
        "-d", str(distance),
        "-bt", box_type,
    ]

    run_cmd(cmd, cwd=cwd)



@task(inp=FILE_IN, top=FILE_INOUT, out=FILE_OUT)
def gmx_solvate_task(gmx, inp, top, out):
    cwd = dirname(out)

    cmd = [
        gmx,
        "solvate",
        "-cp", basename(inp),
        "-cs", "spc216.gro",
        "-o", basename(out),
        "-p", basename(top),
    ]

    run_cmd(cmd, cwd=cwd)



@task(mdp=FILE_IN, conf=FILE_IN, top=FILE_IN, posre=FILE_IN, tpr=FILE_OUT)
def gmx_grompp_task(gmx, mdp, conf, top, posre, tpr, maxwarn):
    cwd = dirname(tpr)

    cmd = [
        gmx,
        "grompp",
        "-f", basename(mdp),
        "-c", basename(conf),
        "-p", basename(top),
        "-o", basename(tpr),
        "-maxwarn", str(maxwarn),
    ]

    run_cmd(cmd, cwd=cwd)



@task(tpr=FILE_IN, top=FILE_INOUT, out=FILE_OUT)
def gmx_genion_task(gmx, tpr, top, out, pname, nname):
    cwd = dirname(out)

    cmd = [
        gmx,
        "genion",
        "-s", basename(tpr),
        "-o", basename(out),
        "-p", basename(top),
        "-pname", pname,
        "-nname", nname,
        "-neutral",
    ]

    # Seleciona o grupo SOL automaticamente.
    run_cmd(cmd, cwd=cwd, stdin_text="SOL\n")



@task(tpr=FILE_IN, confout=FILE_OUT)
def gmx_mdrun_task(gmx_mpi, tpr, deffnm, confout, mpi_np, mpi_runner):
    cwd = dirname(confout)

    if mpi_np > 1:
        cmd = [
            mpi_runner,
            "--allow-run-as-root",
            "-np", str(mpi_np),
            gmx_mpi,
            "mdrun",
            "-s", basename(tpr),
            "-deffnm", basename(deffnm),
        ]
    else:
        cmd = [
            gmx_mpi,
            "mdrun",
            "-s", basename(tpr),
            "-deffnm", basename(deffnm),
        ]

    run_cmd(cmd, cwd=cwd)



def workflow_lysozyme(args):
    os.makedirs(args.workdir, exist_ok=True)

    pdb_src = os.path.abspath(args.pdb)
    pdb_dst = os.path.join(args.workdir, os.path.basename(args.pdb))

    if pdb_src != os.path.abspath(pdb_dst):
        shutil.copy2(pdb_src, pdb_dst)

    ions_mdp, minim_mdp = write_mdp_files(args.workdir)

    processed_gro = os.path.join(args.workdir, "processed.gro")
    topol_top = os.path.join(args.workdir, "topol.top")
    posre_itp = os.path.join(args.workdir, "posre.itp")

    newbox_gro = os.path.join(args.workdir, "newbox.gro")
    solv_gro = os.path.join(args.workdir, "solv.gro")

    ions_tpr = os.path.join(args.workdir, "ions.tpr")
    solv_ions_gro = os.path.join(args.workdir, "solv_ions.gro")

    em_tpr = os.path.join(args.workdir, "em.tpr")
    em_deffnm = os.path.join(args.workdir, "em")
    em_gro = os.path.join(args.workdir, "em.gro")

    gmx_pdb2gmx_task(
        args.gmx,
        pdb_dst,
        processed_gro,
        topol_top,
        posre_itp,
        args.force_field,
        args.water_model,
    )

    gmx_editconf_task(
        args.gmx,
        processed_gro,
        newbox_gro,
        args.box_distance,
        args.box_type,
    )

    gmx_solvate_task(
        args.gmx,
        newbox_gro,
        topol_top,
        solv_gro,
    )

    gmx_grompp_task(
        args.gmx,
        ions_mdp,
        solv_gro,
        topol_top,
        posre_itp,
        ions_tpr,
        args.maxwarn,
    )

    gmx_genion_task(
        args.gmx,
        ions_tpr,
        topol_top,
        solv_ions_gro,
        args.pname,
        args.nname,
    )

    gmx_grompp_task(
        args.gmx,
        minim_mdp,
        solv_ions_gro,
        topol_top,
        posre_itp,
        em_tpr,
        args.maxwarn,
    )

    gmx_mdrun_task(
        args.gmx_mpi,
        em_tpr,
        em_deffnm,
        em_gro,
        args.mpi_np,
        args.mpi_runner,
    )

    compss_barrier()

    print("Workflow Lysozyme finalizado.")
    print("Estrutura minimizada:", em_gro)



def parse_args():
    parser = argparse.ArgumentParser(
        description="Workflows PyCOMPSs: KMeans, Matmul e Lysozyme/GROMACS."
    )

    parser.add_argument(
        "--workflow",
        choices=["kmeans", "matmul", "lysozyme"],
        required=True,
        help="Workflow a ser executado.",
    )

    # KMeans
    parser.add_argument("--points", type=int, default=100000)
    parser.add_argument("--dim", type=int, default=3)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--partitions", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)

    # Matmul
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=512)

    # Lysozyme/GROMACS
    parser.add_argument("--pdb", type=str, default="1aki.pdb")
    parser.add_argument("--workdir", type=str, default="lysozyme_work")
    parser.add_argument("--gmx", type=str, default="gmx")
    parser.add_argument("--gmx-mpi", type=str, default="gmx_mpi")
    parser.add_argument("--mpi-runner", type=str, default="mpirun")
    parser.add_argument("--mpi-np", type=int, default=2)

    parser.add_argument("--force-field", type=str, default="amber99sb-ildn")
    parser.add_argument("--water-model", type=str, default="tip3p")
    parser.add_argument("--box-distance", type=float, default=1.0)
    parser.add_argument("--box-type", type=str, default="cubic")
    parser.add_argument("--pname", type=str, default="NA")
    parser.add_argument("--nname", type=str, default="CL")
    parser.add_argument("--maxwarn", type=int, default=1)

    return parser.parse_args()



def main():
    args = parse_args()

    if args.workflow == "kmeans":
        workflow_kmeans(args)

    elif args.workflow == "matmul":
        workflow_matmul(args)

    elif args.workflow == "lysozyme":
        workflow_lysozyme(args)



if __name__ == "__main__":
    main()
