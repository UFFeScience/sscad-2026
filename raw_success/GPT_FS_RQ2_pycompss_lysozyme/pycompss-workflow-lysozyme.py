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



def run_command(cmd, cwd=None, stdin_text=None):
    """
    Executa um comando externo dentro de uma task.
    """
    print("\n[COMMAND]", " ".join(cmd))
    if cwd is not None:
        print("[CWD]", cwd)

    result = subprocess.run(
        cmd,
        cwd=cwd,
        input=stdin_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )

    print(result.stdout)

    if result.returncode != 0:
        raise RuntimeError(
            "Comando falhou com código %d: %s" %
            (result.returncode, " ".join(cmd))
        )



def write_default_mdp_files(run_dir):
    """
    Cria arquivos .mdp mínimos para genion/grompp e minimização de energia.
    """
    os.makedirs(run_dir, exist_ok=True)

    ions_mdp = os.path.join(run_dir, "ions.mdp")
    minim_mdp = os.path.join(run_dir, "minim.mdp")

    if not os.path.exists(ions_mdp):
        with open(ions_mdp, "w") as f:
            f.write("""\
; ions.mdp
integrator      = steep
emtol           = 1000.0
emstep          = 0.01
nsteps          = 500
cutoff-scheme   = Verlet
nstlist         = 10
rcoulomb        = 1.0
rvdw            = 1.0
coulombtype     = PME
pbc             = xyz
""")

    if not os.path.exists(minim_mdp):
        with open(minim_mdp, "w") as f:
            f.write("""\
; minim.mdp
integrator      = steep
emtol           = 1000.0
emstep          = 0.01
nsteps          = 500
cutoff-scheme   = Verlet
nstlist         = 10
rcoulomb        = 1.0
rvdw            = 1.0
coulombtype     = PME
pbc             = xyz
""")

    return ions_mdp, minim_mdp



@task(pdb=FILE_IN, out_gro=FILE_OUT, top=FILE_OUT, posre=FILE_OUT)
def gmx_pdb2gmx(pdb, out_gro, top, posre, gmx, forcefield, water):
    """
    Gera topologia e estrutura processada usando pdb2gmx.
    """
    cwd = os.path.dirname(os.path.abspath(out_gro))
    os.makedirs(cwd, exist_ok=True)

    cmd = [
        gmx, "pdb2gmx",
        "-f", os.path.abspath(pdb),
        "-o", os.path.basename(out_gro),
        "-p", os.path.basename(top),
        "-i", os.path.basename(posre),
        "-ff", forcefield,
        "-water", water
    ]

    run_command(cmd, cwd=cwd)



@task(in_gro=FILE_IN, out_gro=FILE_OUT)
def gmx_editconf(in_gro, out_gro, gmx, box_distance, box_type):
    """
    Define caixa de simulação.
    """
    cwd = os.path.dirname(os.path.abspath(out_gro))
    os.makedirs(cwd, exist_ok=True)

    cmd = [
        gmx, "editconf",
        "-f", os.path.abspath(in_gro),
        "-o", os.path.basename(out_gro),
        "-c",
        "-d", str(box_distance),
        "-bt", box_type
    ]

    run_command(cmd, cwd=cwd)



@task(in_gro=FILE_IN, top=FILE_INOUT, out_gro=FILE_OUT)
def gmx_solvate(in_gro, top, out_gro, gmx):
    """
    Solvata o sistema e atualiza a topologia.
    """
    cwd = os.path.dirname(os.path.abspath(out_gro))
    os.makedirs(cwd, exist_ok=True)

    cmd = [
        gmx, "solvate",
        "-cp", os.path.abspath(in_gro),
        "-cs", "spc216.gro",
        "-o", os.path.basename(out_gro),
        "-p", os.path.abspath(top)
    ]

    run_command(cmd, cwd=cwd)



@task(mdp=FILE_IN, in_gro=FILE_IN, top=FILE_IN, posre=FILE_IN, out_tpr=FILE_OUT)
def gmx_grompp(mdp, in_gro, top, posre, out_tpr, gmx, maxwarn):
    """
    Executa grompp para gerar arquivo .tpr.
    O parâmetro posre é declarado como dependência de arquivo,
    pois a topologia gerada pelo pdb2gmx normalmente inclui posre.itp.
    """
    _ = posre

    cwd = os.path.dirname(os.path.abspath(out_tpr))
    os.makedirs(cwd, exist_ok=True)

    cmd = [
        gmx, "grompp",
        "-f", os.path.abspath(mdp),
        "-c", os.path.abspath(in_gro),
        "-p", os.path.abspath(top),
        "-o", os.path.basename(out_tpr),
        "-maxwarn", str(maxwarn)
    ]

    run_command(cmd, cwd=cwd)



@task(tpr=FILE_IN, top=FILE_INOUT, out_gro=FILE_OUT)
def gmx_genion(tpr, top, out_gro, gmx, pname, nname, ion_group):
    """
    Adiciona íons ao sistema e atualiza a topologia.
    Normalmente o grupo de solvente é SOL.
    """
    cwd = os.path.dirname(os.path.abspath(out_gro))
    os.makedirs(cwd, exist_ok=True)

    cmd = [
        gmx, "genion",
        "-s", os.path.abspath(tpr),
        "-o", os.path.basename(out_gro),
        "-p", os.path.abspath(top),
        "-pname", pname,
        "-nname", nname,
        "-neutral"
    ]

    run_command(cmd, cwd=cwd, stdin_text=ion_group + "\n")



@task(tpr=FILE_IN, out_gro=FILE_OUT, out_log=FILE_OUT, returns=str)
def gmx_mdrun_energy_minimization(tpr, out_gro, out_log, gmx_mpi, mpi_ranks, ntomp):
    """
    Executa mdrun usando gmx_mpi para minimização de energia.
    """
    cwd = os.path.dirname(os.path.abspath(out_gro))
    os.makedirs(cwd, exist_ok=True)

    deffnm = os.path.splitext(os.path.basename(out_gro))[0]

    if mpi_ranks > 1:
        cmd = [
            "mpirun", "-np", str(mpi_ranks),
            gmx_mpi, "mdrun",
            "-s", os.path.abspath(tpr),
            "-deffnm", deffnm
        ]
    else:
        cmd = [
            gmx_mpi, "mdrun",
            "-s", os.path.abspath(tpr),
            "-deffnm", deffnm
        ]

    if ntomp > 0:
        cmd += ["-ntomp", str(ntomp)]

    run_command(cmd, cwd=cwd)

    return os.path.abspath(out_gro)



def run_lysozyme(args):
    """
    Workflow PyCOMPSs para preparação e minimização de energia
    de uma proteína Lysozyme com GROMACS.
    """
    start = time.time()

    pdb = os.path.abspath(args.pdb)
    run_dir = os.path.abspath(args.run_dir)

    if not os.path.exists(pdb):
        raise FileNotFoundError(
            "Arquivo PDB não encontrado: %s\n"
            "Forneça, por exemplo, um PDB de lisozima como 1AKI.pdb." % pdb
        )

    os.makedirs(run_dir, exist_ok=True)

    ions_mdp, minim_mdp = write_default_mdp_files(run_dir)

    processed_gro = os.path.join(run_dir, "processed.gro")
    topol_top = os.path.join(run_dir, "topol.top")
    posre_itp = os.path.join(run_dir, "posre.itp")

    newbox_gro = os.path.join(run_dir, "newbox.gro")
    solvated_gro = os.path.join(run_dir, "solvated.gro")

    ions_tpr = os.path.join(run_dir, "ions.tpr")
    solv_ions_gro = os.path.join(run_dir, "solv_ions.gro")

    em_tpr = os.path.join(run_dir, "em.tpr")
    em_gro = os.path.join(run_dir, "em.gro")
    em_log = os.path.join(run_dir, "em.log")

    print("Executando workflow Lysozyme/GROMACS")
    print("PDB:", pdb)
    print("Diretório de execução:", run_dir)
    print("gmx:", args.gmx)
    print("gmx_mpi:", args.gmx_mpi)
    print("Force field:", args.forcefield)
    print("Water model:", args.water)

    gmx_pdb2gmx(
        pdb,
        processed_gro,
        topol_top,
        posre_itp,
        args.gmx,
        args.forcefield,
        args.water
    )

    gmx_editconf(
        processed_gro,
        newbox_gro,
        args.gmx,
        args.box_distance,
        args.box_type
    )

    gmx_solvate(
        newbox_gro,
        topol_top,
        solvated_gro,
        args.gmx
    )

    gmx_grompp(
        ions_mdp,
        solvated_gro,
        topol_top,
        posre_itp,
        ions_tpr,
        args.gmx,
        args.maxwarn
    )

    gmx_genion(
        ions_tpr,
        topol_top,
        solv_ions_gro,
        args.gmx,
        args.pname,
        args.nname,
        args.ion_group
    )

    gmx_grompp(
        minim_mdp,
        solv_ions_gro,
        topol_top,
        posre_itp,
        em_tpr,
        args.gmx,
        args.maxwarn
    )

    final_gro = gmx_mdrun_energy_minimization(
        em_tpr,
        em_gro,
        em_log,
        args.gmx_mpi,
        args.mpi_ranks,
        args.ntomp
    )

    final_gro = compss_wait_on(final_gro)

    elapsed = time.time() - start

    print("\nResultado Lysozyme")
    print("Minimização concluída.")
    print("Arquivo final:", final_gro)
    print("Log:", em_log)
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
