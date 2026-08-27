#!/usr/bin/env python3
import os
import argparse

import parsl
from parsl.app.app import python_app, bash_app
from parsl.configs.local_threads import config
from parsl.data_provider.files import File


@python_app
def fetch_lysozyme_pdb(outputs=[]):
    import urllib.request

    url = "https://files.rcsb.org/download/1AKI.pdb"

    with urllib.request.urlopen(url) as response:
        pdb_text = response.read().decode("utf-8")

    cleaned = []

    for line in pdb_text.splitlines():
        if line.startswith("ATOM"):
            cleaned.append(line)
        elif line.startswith("TER"):
            cleaned.append(line)
        elif line.startswith("END"):
            cleaned.append(line)

    with open(outputs[0].filepath, "w") as f:
        f.write("\n".join(cleaned))
        f.write("\n")

    return outputs[0].filepath


@python_app
def write_mdp_files(em_steps, outputs=[]):
    ions_mdp = """
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

    em_mdp = f"""
integrator      = steep
emtol           = 1000.0
emstep          = 0.01
nsteps          = {em_steps}
cutoff-scheme   = Verlet
coulombtype     = PME
rcoulomb        = 1.0
rvdw            = 1.0
pbc             = xyz
"""

    with open(outputs[0].filepath, "w") as f:
        f.write(ions_mdp.strip() + "\n")

    with open(outputs[1].filepath, "w") as f:
        f.write(em_mdp.strip() + "\n")

    return {
        "ions_mdp": outputs[0].filepath,
        "em_mdp": outputs[1].filepath,
    }


@bash_app
def pdb2gmx_app(gmx, forcefield, water_model, inputs=[], outputs=[], stdout=None, stderr=None):
    import shlex

    inp = shlex.quote(inputs[0].filepath)
    gro = shlex.quote(outputs[0].filepath)
    top = shlex.quote(outputs[1].filepath)
    posre = shlex.quote(outputs[2].filepath)

    return f"""
set -euo pipefail

{shlex.quote(gmx)} pdb2gmx \\
    -f {inp} \\
    -o {gro} \\
    -p {top} \\
    -i {posre} \\
    -ff {shlex.quote(forcefield)} \\
    -water {shlex.quote(water_model)} \\
    -ignh
"""


@bash_app
def editconf_app(gmx, distance, box_type, inputs=[], outputs=[], stdout=None, stderr=None):
    import shlex

    return f"""
set -euo pipefail

{shlex.quote(gmx)} editconf \\
    -f {shlex.quote(inputs[0].filepath)} \\
    -o {shlex.quote(outputs[0].filepath)} \\
    -c \\
    -d {distance} \\
    -bt {shlex.quote(box_type)}
"""


@bash_app
def solvate_app(gmx, inputs=[], outputs=[], stdout=None, stderr=None):
    import shlex

    boxed_gro = shlex.quote(inputs[0].filepath)
    topol_in = shlex.quote(inputs[1].filepath)

    solvated_gro = shlex.quote(outputs[0].filepath)
    topol_solv = shlex.quote(outputs[1].filepath)

    return f"""
set -euo pipefail

cp {topol_in} {topol_solv}

{shlex.quote(gmx)} solvate \\
    -cp {boxed_gro} \\
    -cs spc216.gro \\
    -o {solvated_gro} \\
    -p {topol_solv}
"""


@bash_app
def grompp_ions_app(gmx, maxwarn, inputs=[], outputs=[], stdout=None, stderr=None):
    import shlex

    ions_mdp = shlex.quote(inputs[0].filepath)
    solvated_gro = shlex.quote(inputs[1].filepath)
    topol_solv = shlex.quote(inputs[2].filepath)

    ions_tpr = shlex.quote(outputs[0].filepath)

    return f"""
set -euo pipefail

{shlex.quote(gmx)} grompp \\
    -f {ions_mdp} \\
    -c {solvated_gro} \\
    -p {topol_solv} \\
    -o {ions_tpr} \\
    -maxwarn {maxwarn}
"""


@bash_app
def genion_app(gmx, ion_group, inputs=[], outputs=[], stdout=None, stderr=None):
    import shlex

    ions_tpr = shlex.quote(inputs[0].filepath)
    topol_solv = shlex.quote(inputs[1].filepath)

    solv_ions_gro = shlex.quote(outputs[0].filepath)
    topol_ions = shlex.quote(outputs[1].filepath)

    return f"""
set -euo pipefail

cp {topol_solv} {topol_ions}

printf "{ion_group}\\n" | {shlex.quote(gmx)} genion \\
    -s {ions_tpr} \\
    -o {solv_ions_gro} \\
    -p {topol_ions} \\
    -pname NA \\
    -nname CL \\
    -neutral
"""


@bash_app
def grompp_em_app(gmx, maxwarn, inputs=[], outputs=[], stdout=None, stderr=None):
    import shlex

    em_mdp = shlex.quote(inputs[0].filepath)
    solv_ions_gro = shlex.quote(inputs[1].filepath)
    topol_ions = shlex.quote(inputs[2].filepath)

    em_tpr = shlex.quote(outputs[0].filepath)

    return f"""
set -euo pipefail

{shlex.quote(gmx)} grompp \\
    -f {em_mdp} \\
    -c {solv_ions_gro} \\
    -p {topol_ions} \\
    -o {em_tpr} \\
    -maxwarn {maxwarn}
"""


@bash_app
def mdrun_em_app(gmx_mpi, launcher, nproc, inputs=[], outputs=[], stdout=None, stderr=None):
    import os
    import shlex

    em_tpr = inputs[0].filepath
    prefix = outputs[0].filepath[:-4] if outputs[0].filepath.endswith(".gro") else os.path.join(
        os.path.dirname(outputs[0].filepath), "em"
    )

    q_gmx_mpi = shlex.quote(gmx_mpi)
    q_prefix = shlex.quote(prefix)
    q_em_tpr = shlex.quote(em_tpr)

    if launcher == "none":
        run_cmd = f"{q_gmx_mpi} mdrun -s {q_em_tpr} -deffnm {q_prefix}"
    elif os.path.basename(launcher) == "srun":
        run_cmd = f"{shlex.quote(launcher)} -n {nproc} {q_gmx_mpi} mdrun -s {q_em_tpr} -deffnm {q_prefix}"
    else:
        run_cmd = f"{shlex.quote(launcher)} -np {nproc} {q_gmx_mpi} mdrun -s {q_em_tpr} -deffnm {q_prefix}"

    return f"""
set -euo pipefail

{run_cmd}

test -f {shlex.quote(outputs[0].filepath)}
test -f {shlex.quote(outputs[1].filepath)}
test -f {shlex.quote(outputs[2].filepath)}
"""


def main():
    parser = argparse.ArgumentParser(
        description="Workflow Parsl para preparação e minimização de energia de Lysozyme com GROMACS."
    )

    parser.add_argument(
        "--pdb",
        default=None,
        help="Arquivo PDB de entrada. Se omitido, o script baixa automaticamente o PDB 1AKI.",
    )

    parser.add_argument("--gmx", default="gmx", help="Executável serial do GROMACS.")
    parser.add_argument("--gmx-mpi", default="gmx_mpi", help="Executável MPI do GROMACS.")
    parser.add_argument("--launcher", default="mpirun", help="mpirun, srun ou none.")
    parser.add_argument("--nproc", type=int, default=2)

    parser.add_argument("--forcefield", default="amber99sb-ildn")
    parser.add_argument("--water", default="tip3p")
    parser.add_argument("--distance", type=float, default=1.0)
    parser.add_argument("--box-type", default="cubic")
    parser.add_argument("--em-steps", type=int, default=500)
    parser.add_argument("--ion-group", default="SOL")
    parser.add_argument("--maxwarn", type=int, default=1)

    parser.add_argument("--outdir", default="output_lysozyme")

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    parsl.load(config)

    try:
        outdir = os.path.abspath(args.outdir)

        if args.pdb is None:
            pdb_file = File(os.path.join(outdir, "1AKI_clean.pdb"))
            pdb_future = fetch_lysozyme_pdb(outputs=[pdb_file])
            pdb_input = pdb_future.outputs[0]
        else:
            pdb_input = File(os.path.abspath(args.pdb))

        ions_mdp = File(os.path.join(outdir, "ions.mdp"))
        em_mdp = File(os.path.join(outdir, "em.mdp"))

        mdp_future = write_mdp_files(
            args.em_steps,
            outputs=[ions_mdp, em_mdp],
        )

        processed_gro = File(os.path.join(outdir, "processed.gro"))
        topol_raw = File(os.path.join(outdir, "topol_raw.top"))
        posre = File(os.path.join(outdir, "posre.itp"))

        pdb2gmx = pdb2gmx_app(
            args.gmx,
            args.forcefield,
            args.water,
            inputs=[pdb_input],
            outputs=[processed_gro, topol_raw, posre],
            stdout=(os.path.join(outdir, "01_pdb2gmx.out"), "w"),
            stderr=(os.path.join(outdir, "01_pdb2gmx.err"), "w"),
        )

        boxed_gro = File(os.path.join(outdir, "boxed.gro"))

        editconf = editconf_app(
            args.gmx,
            args.distance,
            args.box_type,
            inputs=[pdb2gmx.outputs[0]],
            outputs=[boxed_gro],
            stdout=(os.path.join(outdir, "02_editconf.out"), "w"),
            stderr=(os.path.join(outdir, "02_editconf.err"), "w"),
        )

        solvated_gro = File(os.path.join(outdir, "solvated.gro"))
        topol_solv = File(os.path.join(outdir, "topol_solv.top"))

        solvate = solvate_app(
            args.gmx,
            inputs=[editconf.outputs[0], pdb2gmx.outputs[1]],
            outputs=[solvated_gro, topol_solv],
            stdout=(os.path.join(outdir, "03_solvate.out"), "w"),
            stderr=(os.path.join(outdir, "03_solvate.err"), "w"),
        )

        ions_tpr = File(os.path.join(outdir, "ions.tpr"))

        grompp_ions = grompp_ions_app(
            args.gmx,
            args.maxwarn,
            inputs=[mdp_future.outputs[0], solvate.outputs[0], solvate.outputs[1]],
            outputs=[ions_tpr],
            stdout=(os.path.join(outdir, "04_grompp_ions.out"), "w"),
            stderr=(os.path.join(outdir, "04_grompp_ions.err"), "w"),
        )

        solv_ions_gro = File(os.path.join(outdir, "solv_ions.gro"))
        topol_ions = File(os.path.join(outdir, "topol_ions.top"))

        genion = genion_app(
            args.gmx,
            args.ion_group,
            inputs=[grompp_ions.outputs[0], solvate.outputs[1]],
            outputs=[solv_ions_gro, topol_ions],
            stdout=(os.path.join(outdir, "05_genion.out"), "w"),
            stderr=(os.path.join(outdir, "05_genion.err"), "w"),
        )

        em_tpr = File(os.path.join(outdir, "em.tpr"))

        grompp_em = grompp_em_app(
            args.gmx,
            args.maxwarn,
            inputs=[mdp_future.outputs[1], genion.outputs[0], genion.outputs[1]],
            outputs=[em_tpr],
            stdout=(os.path.join(outdir, "06_grompp_em.out"), "w"),
            stderr=(os.path.join(outdir, "06_grompp_em.err"), "w"),
        )

        em_gro = File(os.path.join(outdir, "em.gro"))
        em_edr = File(os.path.join(outdir, "em.edr"))
        em_log = File(os.path.join(outdir, "em.log"))

        mdrun = mdrun_em_app(
            args.gmx_mpi,
            args.launcher,
            args.nproc,
            inputs=[grompp_em.outputs[0]],
            outputs=[em_gro, em_edr, em_log],
            stdout=(os.path.join(outdir, "07_mdrun_em.out"), "w"),
            stderr=(os.path.join(outdir, "07_mdrun_em.err"), "w"),
        )

        mdrun.result()

        print("Workflow Lysozyme finalizado com sucesso.")
        print(f"Estrutura minimizada: {em_gro.filepath}")
        print(f"Arquivo de energia:   {em_edr.filepath}")
        print(f"Log do GROMACS:       {em_log.filepath}")

    finally:
        parsl.clear()


if __name__ == "__main__":
    main()