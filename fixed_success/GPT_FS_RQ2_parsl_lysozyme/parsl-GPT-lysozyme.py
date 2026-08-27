#!/usr/bin/env python3
import os
import argparse
from pathlib import Path

import parsl
from parsl.app.app import bash_app
from parsl.configs.local_threads import config
from parsl.data_provider.files import File


@bash_app
def check_gromacs(gmx, gmx_mpi, outputs=[], stdout=None, stderr=None):
    return f"""
set -euo pipefail

command -v {gmx}
command -v {gmx_mpi}

echo "=== gmx ===" > {outputs[0].filepath}
{gmx} --version >> {outputs[0].filepath}

echo "" >> {outputs[0].filepath}
echo "=== gmx_mpi ===" >> {outputs[0].filepath}
{gmx_mpi} --version >> {outputs[0].filepath}
"""


@bash_app
def prepare_pdb(pdb_source, workdir, inputs=[], outputs=[], stdout=None, stderr=None):
    return f"""
set -euo pipefail

mkdir -p {workdir}
cd {workdir}

if [ -n "{pdb_source}" ]; then
    cp "{pdb_source}" raw_lysozyme.pdb
else
    if command -v curl >/dev/null 2>&1; then
        curl -L -o raw_lysozyme.pdb https://files.rcsb.org/download/1AKI.pdb
    elif command -v wget >/dev/null 2>&1; then
        wget -O raw_lysozyme.pdb https://files.rcsb.org/download/1AKI.pdb
    else
        echo "curl ou wget é necessário para baixar o PDB 1AKI." >&2
        exit 1
    fi
fi

grep -v '^HETATM' raw_lysozyme.pdb | grep -v '^CONECT' > {outputs[0].filepath}
"""


@bash_app
def pdb2gmx_app(gmx, workdir, forcefield, water, inputs=[], outputs=[], stdout=None, stderr=None):
    return f"""
set -euo pipefail

cd {workdir}

{gmx} pdb2gmx \\
    -f {inputs[0].filepath} \\
    -o {outputs[0].filepath} \\
    -p {outputs[1].filepath} \\
    -i {outputs[2].filepath} \\
    -ff {forcefield} \\
    -water {water} \\
    -ignh
"""


@bash_app
def editconf_app(gmx, workdir, inputs=[], outputs=[], stdout=None, stderr=None):
    return f"""
set -euo pipefail

cd {workdir}

{gmx} editconf \\
    -f {inputs[0].filepath} \\
    -o {outputs[0].filepath} \\
    -c \\
    -d 1.0 \\
    -bt cubic
"""


@bash_app
def solvate_app(gmx, workdir, topol, inputs=[], outputs=[], stdout=None, stderr=None):
    return f"""
set -euo pipefail

cd {workdir}

{gmx} solvate \\
    -cp {inputs[0].filepath} \\
    -cs spc216.gro \\
    -o {outputs[0].filepath} \\
    -p {topol}
"""


@bash_app
def grompp_ions_app(gmx, workdir, topol, inputs=[], outputs=[], stdout=None, stderr=None):
    return f"""
set -euo pipefail

cd {workdir}

cat > ions.mdp << 'EOF'
integrator  = steep
emtol       = 1000.0
emstep      = 0.01
nsteps      = 500
nstlist     = 10
cutoff-scheme = Verlet
coulombtype = PME
rcoulomb    = 1.0
rvdw        = 1.0
pbc         = xyz
EOF

{gmx} grompp \\
    -f ions.mdp \\
    -c {inputs[0].filepath} \\
    -p {topol} \\
    -o {outputs[0].filepath} \\
    -maxwarn 2
"""


@bash_app
def genion_app(gmx, workdir, topol, inputs=[], outputs=[], stdout=None, stderr=None):
    return f"""
set -euo pipefail

cd {workdir}

printf "SOL\\n" | {gmx} genion \\
    -s {inputs[0].filepath} \\
    -o {outputs[0].filepath} \\
    -p {topol} \\
    -pname NA \\
    -nname CL \\
    -neutral
"""


@bash_app
def grompp_minim_app(gmx, workdir, topol, inputs=[], outputs=[], stdout=None, stderr=None):
    return f"""
set -euo pipefail

cd {workdir}

cat > minim.mdp << 'EOF'
integrator      = steep
emtol           = 1000.0
emstep          = 0.01
nsteps          = 50000

nstlist         = 10
cutoff-scheme   = Verlet
coulombtype     = PME
rcoulomb        = 1.0
rvdw            = 1.0
pbc             = xyz

constraints     = none
EOF

{gmx} grompp \\
    -f minim.mdp \\
    -c {inputs[0].filepath} \\
    -p {topol} \\
    -o {outputs[0].filepath} \\
    -maxwarn 2
"""


@bash_app
def mdrun_minim_app(gmx_mpi, mpi_launcher, workdir, inputs=[], outputs=[], stdout=None, stderr=None):
    return f"""
set -euo pipefail

cd {workdir}

{mpi_launcher} {gmx_mpi} mdrun \\
    -deffnm em

test em.gro -ef {outputs[0].filepath} || cp em.gro {outputs[0].filepath}
test em.edr -ef {outputs[1].filepath} || cp em.edr {outputs[1].filepath}
test em.log -ef {outputs[2].filepath} || cp em.log {outputs[2].filepath}
"""


def main():
    parser = argparse.ArgumentParser(
        description="Workflow Parsl para preparação e minimização de Lysozyme com GROMACS."
    )

    parser.add_argument(
        "--workdir",
        default="output/lysozyme",
        help="Diretório de trabalho do workflow."
    )

    parser.add_argument(
        "--pdb",
        default="",
        help="Arquivo PDB de entrada. Se omitido, baixa o PDB 1AKI."
    )

    parser.add_argument(
        "--gmx",
        default="gmx",
        help="Executável serial do GROMACS."
    )

    parser.add_argument(
        "--gmx-mpi",
        default="gmx_mpi",
        help="Executável MPI do GROMACS."
    )

    parser.add_argument(
        "--mpi-launcher",
        default=os.environ.get("MPI_LAUNCHER", "mpirun -np 2"),
        help='Lançador MPI. Exemplo: "mpirun -np 4" ou "srun -n 4".'
    )

    parser.add_argument(
        "--forcefield",
        default="amber99sb-ildn",
        help="Campo de força usado pelo pdb2gmx."
    )

    parser.add_argument(
        "--water",
        default="spce",
        help="Modelo de água usado pelo pdb2gmx."
    )

    args = parser.parse_args()

    workdir = Path(args.workdir).resolve()
    logs = workdir / "logs"

    os.makedirs(workdir, exist_ok=True)
    os.makedirs(logs, exist_ok=True)

    parsl.load(config)

    try:
        version_file = File(str(workdir / "gromacs_version.txt"))

        check = check_gromacs(
            args.gmx,
            args.gmx_mpi,
            outputs=[version_file],
            stdout=str(logs / "check_gromacs.out"),
            stderr=str(logs / "check_gromacs.err"),
        )

        clean_pdb = File(str(workdir / "lysozyme_clean.pdb"))

        prep = prepare_pdb(
            args.pdb,
            str(workdir),
            inputs=[check.outputs[0]],
            outputs=[clean_pdb],
            stdout=str(logs / "prepare_pdb.out"),
            stderr=str(logs / "prepare_pdb.err"),
        )

        processed_gro = File(str(workdir / "lysozyme_processed.gro"))
        topol_top = File(str(workdir / "topol.top"))
        posre_itp = File(str(workdir / "posre.itp"))

        p2g = pdb2gmx_app(
            args.gmx,
            str(workdir),
            args.forcefield,
            args.water,
            inputs=[prep.outputs[0]],
            outputs=[processed_gro, topol_top, posre_itp],
            stdout=str(logs / "pdb2gmx.out"),
            stderr=str(logs / "pdb2gmx.err"),
        )

        newbox_gro = File(str(workdir / "lysozyme_newbox.gro"))

        box = editconf_app(
            args.gmx,
            str(workdir),
            inputs=[p2g.outputs[0]],
            outputs=[newbox_gro],
            stdout=str(logs / "editconf.out"),
            stderr=str(logs / "editconf.err"),
        )

        solv_gro = File(str(workdir / "lysozyme_solv.gro"))

        solv = solvate_app(
            args.gmx,
            str(workdir),
            str(workdir / "topol.top"),
            inputs=[box.outputs[0], p2g.outputs[1]],
            outputs=[solv_gro],
            stdout=str(logs / "solvate.out"),
            stderr=str(logs / "solvate.err"),
        )

        ions_tpr = File(str(workdir / "ions.tpr"))

        ions_tpr_future = grompp_ions_app(
            args.gmx,
            str(workdir),
            str(workdir / "topol.top"),
            inputs=[solv.outputs[0]],
            outputs=[ions_tpr],
            stdout=str(logs / "grompp_ions.out"),
            stderr=str(logs / "grompp_ions.err"),
        )

        solv_ions_gro = File(str(workdir / "lysozyme_solv_ions.gro"))

        ions = genion_app(
            args.gmx,
            str(workdir),
            str(workdir / "topol.top"),
            inputs=[ions_tpr_future.outputs[0]],
            outputs=[solv_ions_gro],
            stdout=str(logs / "genion.out"),
            stderr=str(logs / "genion.err"),
        )

        em_tpr = File(str(workdir / "em.tpr"))

        em_tpr_future = grompp_minim_app(
            args.gmx,
            str(workdir),
            str(workdir / "topol.top"),
            inputs=[ions.outputs[0]],
            outputs=[em_tpr],
            stdout=str(logs / "grompp_minim.out"),
            stderr=str(logs / "grompp_minim.err"),
        )

        em_gro = File(str(workdir / "em.gro"))
        em_edr = File(str(workdir / "em.edr"))
        em_log = File(str(workdir / "em.log"))

        em = mdrun_minim_app(
            args.gmx_mpi,
            args.mpi_launcher,
            str(workdir),
            inputs=[em_tpr_future.outputs[0]],
            outputs=[em_gro, em_edr, em_log],
            stdout=str(logs / "mdrun_minim.out"),
            stderr=str(logs / "mdrun_minim.err"),
        )

        final_gro = em.outputs[0].result()

        print("Workflow Lysozyme finalizado.")
        print(f"Estrutura minimizada: {final_gro}")
        print(f"Diretório de trabalho: {workdir}")
        print(f"Logs: {logs}")

    finally:
        parsl.clear()


if __name__ == "__main__":
    main()