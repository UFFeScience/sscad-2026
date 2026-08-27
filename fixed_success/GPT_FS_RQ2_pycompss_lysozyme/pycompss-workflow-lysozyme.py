import os



import time



import shlex



import argparse



import subprocess



import urllib.request



from pycompss.api.task import task



from pycompss.api.api import compss_wait_on



def mkdir(path):
    os.makedirs(path, exist_ok=True)
    return os.path.abspath(path)



def write_gromacs_mdp_files(workdir):
    """
    Cria arquivos MDP mínimos para preparação de íons e minimização de energia.
    """
    mkdir(workdir)

    ions_mdp = os.path.join(workdir, "ions.mdp")
    minim_mdp = os.path.join(workdir, "minim.mdp")

    ions_content = """\
integrator      = steep
emtol           = 1000.0
emstep          = 0.01
nsteps          = 500
nstlist         = 1
cutoff-scheme   = Verlet
coulombtype     = Cut-off
rcoulomb        = 1.0
rvdw            = 1.0
pbc             = xyz
"""

    minim_content = """\
integrator      = steep
emtol           = 1000.0
emstep          = 0.01
nsteps          = 5000
nstenergy       = 10
cutoff-scheme   = Verlet
coulombtype     = PME
rcoulomb        = 1.0
rvdw            = 1.0
pbc             = xyz
"""

    with open(ions_mdp, "w") as f:
        f.write(ions_content)

    with open(minim_mdp, "w") as f:
        f.write(minim_content)

    return ions_mdp, minim_mdp



def run_command(cmd, cwd, stdin_text=None):
    """
    Executa comando externo dentro de uma task.
    """
    print("Executando em {}: {}".format(cwd, " ".join(cmd)))

    proc = subprocess.run(
        cmd,
        cwd=cwd,
        input=stdin_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    if proc.returncode != 0:
        raise RuntimeError(
            "Comando falhou com código {}:\n{}\n\nSTDOUT:\n{}\n\nSTDERR:\n{}".format(
                proc.returncode,
                " ".join(cmd),
                proc.stdout,
                proc.stderr
            )
        )

    return proc.stdout, proc.stderr



@task(returns=1)
def download_lysozyme_pdb(workdir, pdb_id):
    """
    Baixa um PDB do RCSB. Por padrão, usa 1AKI, lisozima.
    """
    mkdir(workdir)

    pdb_id = pdb_id.upper()
    url = "https://files.rcsb.org/download/{}.pdb".format(pdb_id)
    out_pdb = os.path.join(workdir, "{}.pdb".format(pdb_id))

    print("Baixando PDB de {}".format(url))
    urllib.request.urlretrieve(url, out_pdb)

    return out_pdb



@task(returns=1)
def lysozyme_pdb2gmx(input_pdb, workdir, gmx_bin, forcefield, water_model):
    """
    Gera topologia usando GROMACS pdb2gmx.
    Essa etapa usa explicitamente a topologia pdb2gmx.
    """
    mkdir(workdir)

    processed_gro = os.path.join(workdir, "processed.gro")
    topol_top = os.path.join(workdir, "topol.top")

    cmd = [
        gmx_bin,
        "pdb2gmx",
        "-f", input_pdb,
        "-o", processed_gro,
        "-p", topol_top,
        "-ff", forcefield,
        "-water", water_model,
        "-ignh"
    ]

    run_command(cmd, workdir)

    return {
        "gro": processed_gro,
        "top": topol_top,
        "workdir": workdir
    }



@task(returns=1)
def lysozyme_editconf(state, gmx_bin, box_distance, box_type):
    """
    Define caixa de simulação.
    """
    workdir = state["workdir"]
    newbox_gro = os.path.join(workdir, "newbox.gro")

    cmd = [
        gmx_bin,
        "editconf",
        "-f", state["gro"],
        "-o", newbox_gro,
        "-c",
        "-d", str(box_distance),
        "-bt", box_type
    ]

    run_command(cmd, workdir)

    state["gro"] = newbox_gro
    return state



@task(returns=1)
def lysozyme_solvate(state, gmx_bin):
    """
    Solvata a proteína e atualiza topol.top.
    """
    workdir = state["workdir"]
    solv_gro = os.path.join(workdir, "solv.gro")

    cmd = [
        gmx_bin,
        "solvate",
        "-cp", state["gro"],
        "-cs", "spc216.gro",
        "-o", solv_gro,
        "-p", state["top"]
    ]

    run_command(cmd, workdir)

    state["gro"] = solv_gro
    return state



@task(returns=1)
def lysozyme_grompp_ions(state, gmx_bin, ions_mdp):
    """
    Pré-processamento para adição de íons.
    """
    workdir = state["workdir"]
    ions_tpr = os.path.join(workdir, "ions.tpr")

    cmd = [
        gmx_bin,
        "grompp",
        "-f", ions_mdp,
        "-c", state["gro"],
        "-p", state["top"],
        "-o", ions_tpr,
        "-maxwarn", "2"
    ]

    run_command(cmd, workdir)

    state["tpr"] = ions_tpr
    return state



@task(returns=1)
def lysozyme_genion(state, gmx_bin, pname, nname):
    """
    Adiciona íons. O GROMACS pergunta qual grupo substituir;
    normalmente 'SOL' é o grupo da água.
    """
    workdir = state["workdir"]
    solv_ions_gro = os.path.join(workdir, "solv_ions.gro")

    cmd = [
        gmx_bin,
        "genion",
        "-s", state["tpr"],
        "-o", solv_ions_gro,
        "-p", state["top"],
        "-pname", pname,
        "-nname", nname,
        "-neutral"
    ]

    run_command(cmd, workdir, stdin_text="SOL\n")

    state["gro"] = solv_ions_gro
    return state



@task(returns=1)
def lysozyme_grompp_minimization(state, gmx_bin, minim_mdp):
    """
    Pré-processamento da minimização de energia.
    """
    workdir = state["workdir"]
    em_tpr = os.path.join(workdir, "em.tpr")

    cmd = [
        gmx_bin,
        "grompp",
        "-f", minim_mdp,
        "-c", state["gro"],
        "-p", state["top"],
        "-o", em_tpr,
        "-maxwarn", "2"
    ]

    run_command(cmd, workdir)

    state["tpr"] = em_tpr
    state["deffnm"] = os.path.join(workdir, "em")
    return state



@task(returns=1)
def lysozyme_mdrun(state, gmx_mpi_bin, mpi_runner, mdrun_args):
    """
    Executa a minimização com gmx_mpi mdrun.
    Se mpi_runner for informado, exemplo: 'mpirun -np 4'.
    """
    workdir = state["workdir"]

    if mpi_runner:
        cmd = shlex.split(mpi_runner) + [
            "--allow-run-as-root",
            gmx_mpi_bin,
            "mdrun",
            "-deffnm", state["deffnm"]
        ]
    else:
        cmd = [
            gmx_mpi_bin,
            "mdrun",
            "-deffnm", state["deffnm"]
        ]

    if mdrun_args:
        cmd.extend(shlex.split(mdrun_args))

    run_command(cmd, workdir)

    result = {
        "workdir": workdir,
        "gro": os.path.join(workdir, "em.gro"),
        "log": os.path.join(workdir, "em.log"),
        "edr": os.path.join(workdir, "em.edr"),
        "trr": os.path.join(workdir, "em.trr"),
        "top": state["top"]
    }

    return result



def run_lysozyme(args):
    print("\n=== Workflow Lysozyme / GROMACS ===")
    start = time.time()

    workdir = mkdir(args.lyso_workdir)
    ions_mdp, minim_mdp = write_gromacs_mdp_files(workdir)

    if args.lyso_pdb:
        input_pdb = os.path.abspath(args.lyso_pdb)
    else:
        input_pdb = download_lysozyme_pdb(workdir, args.lyso_pdb_id)

    state = lysozyme_pdb2gmx(
        input_pdb,
        workdir,
        args.gmx,
        args.lyso_forcefield,
        args.lyso_water
    )

    state = lysozyme_editconf(
        state,
        args.gmx,
        args.lyso_box_distance,
        args.lyso_box_type
    )

    state = lysozyme_solvate(state, args.gmx)

    state = lysozyme_grompp_ions(
        state,
        args.gmx,
        ions_mdp
    )

    state = lysozyme_genion(
        state,
        args.gmx,
        args.lyso_pname,
        args.lyso_nname
    )

    state = lysozyme_grompp_minimization(
        state,
        args.gmx,
        minim_mdp
    )

    result = lysozyme_mdrun(
        state,
        args.gmx_mpi,
        args.mpi_runner,
        args.mdrun_args
    )

    result = compss_wait_on(result)

    elapsed = time.time() - start

    print("\nArquivos gerados:")
    for key, value in result.items():
        print("{}: {}".format(key, value))

    print("Tempo total Lysozyme/GROMACS: {:.3f}s".format(elapsed))



def parse_args():
    parser = argparse.ArgumentParser(
        description="Workflows PyCOMPSs para KMeans, Matmul e Lysozyme/GROMACS."
    )

    parser.add_argument(
        "--workflow",
        choices=["kmeans", "matmul", "lysozyme", "all"],
        default="kmeans",
        help="Workflow a executar."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed global."
    )

    # KMeans
    parser.add_argument("--kmeans-points", type=int, default=100000)
    parser.add_argument("--kmeans-dim", type=int, default=2)
    parser.add_argument("--kmeans-k", type=int, default=4)
    parser.add_argument("--kmeans-chunks", type=int, default=8)
    parser.add_argument("--kmeans-iters", type=int, default=20)
    parser.add_argument("--kmeans-tol", type=float, default=1e-4)
    parser.add_argument("--kmeans-std", type=float, default=0.8)

    # Matmul
    parser.add_argument("--matmul-size", type=int, default=1024)
    parser.add_argument("--matmul-blocks", type=int, default=4)

    # Lysozyme / GROMACS
    parser.add_argument(
        "--lyso-workdir",
        type=str,
        default=os.path.join(os.getcwd(), "lysozyme_workdir")
    )
    parser.add_argument(
        "--lyso-pdb",
        type=str,
        default=None,
        help="Caminho para um PDB local. Se omitido, baixa o PDB informado em --lyso-pdb-id."
    )
    parser.add_argument(
        "--lyso-pdb-id",
        type=str,
        default="1AKI",
        help="ID PDB a baixar caso --lyso-pdb não seja informado."
    )
    parser.add_argument(
        "--gmx",
        type=str,
        default="gmx",
        help="Binário gmx do GROMACS."
    )
    parser.add_argument(
        "--gmx-mpi",
        type=str,
        default="gmx_mpi",
        help="Binário gmx_mpi do GROMACS."
    )
    parser.add_argument(
        "--mpi-runner",
        type=str,
        default="",
        help="Opcional. Exemplo: 'mpirun -np 4'."
    )
    parser.add_argument(
        "--mdrun-args",
        type=str,
        default="",
        help="Argumentos extras para mdrun. Exemplo: '-ntomp 2'."
    )
    parser.add_argument(
        "--lyso-forcefield",
        type=str,
        default="oplsaa",
        help="Campo de força usado pelo pdb2gmx."
    )
    parser.add_argument(
        "--lyso-water",
        type=str,
        default="spce",
        help="Modelo de água usado pelo pdb2gmx."
    )
    parser.add_argument("--lyso-box-distance", type=float, default=1.0)
    parser.add_argument("--lyso-box-type", type=str, default="cubic")
    parser.add_argument("--lyso-pname", type=str, default="NA")
    parser.add_argument("--lyso-nname", type=str, default="CL")

    return parser.parse_args()



def main():
    args = parse_args()

    if args.workflow in ["kmeans", "all"]:
        run_kmeans(args)

    if args.workflow in ["matmul", "all"]:
        run_matmul(args)

    if args.workflow in ["lysozyme", "all"]:
        run_lysozyme(args)



if __name__ == "__main__":
    main()
