import os
import sys
from os import listdir
from os.path import isfile, join

from parsl import bash_app
import parsl
from parsl.config import Config
from parsl.executors.threads import ThreadPoolExecutor
from parsl.data_provider.files import File

parsl.load(Config(executors=[ThreadPoolExecutor(max_threads=4)]))

GMX_BIN_DIR = os.environ.get("GMX_BIN", "")
GMX = os.path.join(GMX_BIN_DIR, "gmx") if GMX_BIN_DIR else "gmx"
GMX_MPI = os.path.join(GMX_BIN_DIR, "gmx_mpi") if GMX_BIN_DIR else "gmx_mpi"


@bash_app
def generate_topology(protein, structure, topology, inputs=[], outputs=[], stdout=None, stderr=None):
    return (
        f'"{GMX}" pdb2gmx -f "{protein}" -o "{structure}" -p "{topology}" '
        f'-ignh -ff oplsaa -water spce'
    )


@bash_app
def define_box(structure, structure_newbox, inputs=[], outputs=[], stdout=None, stderr=None):
    return (
        f'"{GMX}" editconf -f "{structure}" -o "{structure_newbox}" '
        f'-c -d 1.0 -bt cubic'
    )


@bash_app
def add_solvate(structure_newbox, protein_solv, topology, inputs=[], outputs=[], stdout=None, stderr=None):
    return (
        f'"{GMX}" solvate -cp "{structure_newbox}" -cs spc216.gro '
        f'-o "{protein_solv}" -p "{topology}"'
    )


@bash_app
def assemble_tpr(conf, protein_solv, topology, output, inputs=[], outputs=[], stdout=None, stderr=None):
    return (
        f'"{GMX}" grompp -f "{conf}" -c "{protein_solv}" '
        f'-p "{topology}" -o "{output}"'
    )


@bash_app
def replace_solvent_with_ions(ions, output, topology, group, inputs=[], outputs=[], stdout=None, stderr=None):
    return (
        f'"{GMX}" genion -s "{ions}" -o "{output}" -p "{topology}" '
        f'-pname NA -nname CL -neutral < "{group}"'
    )


@bash_app
def energy_minimization(em, em_energy, inputs=[], outputs=[], stdout=None, stderr=None):
    return (
        f'mpirun -n 1 "{GMX_MPI}" mdrun -v -ntomp 0 -s "{em}" -e "{em_energy}"'
    )


@bash_app
def energy_analisis(em, output, selection, inputs=[], outputs=[], stdout=None, stderr=None):
    return (
        f'"{GMX}" energy -f "{em}" -o "{output}" < "{selection}"'
    )


def main(dataset_path, output_path, config_path):
    print("Starting demo")

    os.makedirs(output_path, exist_ok=True)

    protein_names = []
    protein_pdbs = []

    for f in listdir(dataset_path):
        if isfile(join(dataset_path, f)):
            protein_names.append(f.split('.')[0])
            protein_pdbs.append(join(dataset_path, f))

    proteins = zip(protein_names, protein_pdbs)

    for name, pdb in proteins:
        # 1st step
        structure = join(output_path, name + '.gro')
        topology = join(output_path, name + '.top')
        gen_fut = generate_topology(
            protein=pdb,
            structure=structure,
            topology=topology,
            inputs=[pdb],
            outputs=[File(structure), File(topology)],
            stdout=join(output_path, name + '_pdb2gmx.out'),
            stderr=join(output_path, name + '_pdb2gmx.err'),
        )

        # 2nd step
        structure_newbox = join(output_path, name + '_newbox.gro')
        box_fut = define_box(
            structure=structure,
            structure_newbox=structure_newbox,
            inputs=[gen_fut],
            outputs=[File(structure_newbox)],
            stdout=join(output_path, name + '_editconf.out'),
            stderr=join(output_path, name + '_editconf.err'),
        )

        # 3rd step
        protein_solv = join(output_path, name + '_solv.gro')
        solv_fut = add_solvate(
            structure_newbox=structure_newbox,
            protein_solv=protein_solv,
            topology=topology,
            inputs=[box_fut],
            outputs=[File(protein_solv)],
            stdout=join(output_path, name + '_solvate.out'),
            stderr=join(output_path, name + '_solvate.err'),
        )

        # 4th step - ions
        ions_conf = join(config_path, 'ions.mdp')
        ions = join(output_path, name + '_ions.tpr')
        ions_tpr_fut = assemble_tpr(
            conf=ions_conf,
            protein_solv=protein_solv,
            topology=topology,
            output=ions,
            inputs=[solv_fut],
            outputs=[File(ions)],
            stdout=join(output_path, name + '_grompp_ions.out'),
            stderr=join(output_path, name + '_grompp_ions.err'),
        )

        protein_solv_ions = join(output_path, name + '_solv_ions.gro')
        group = join(config_path, 'genion.group')
        genion_fut = replace_solvent_with_ions(
            ions=ions,
            output=protein_solv_ions,
            topology=topology,
            group=group,
            inputs=[ions_tpr_fut],
            outputs=[File(protein_solv_ions)],
            stdout=join(output_path, name + '_genion.out'),
            stderr=join(output_path, name + '_genion.err'),
        )

        # 5th step - energy minimization
        minim_conf = join(config_path, 'minim.mdp')
        em = join(output_path, name + '_em.tpr')
        em_tpr_fut = assemble_tpr(
            conf=minim_conf,
            protein_solv=protein_solv_ions,
            topology=topology,
            output=em,
            inputs=[genion_fut],
            outputs=[File(em)],
            stdout=join(output_path, name + '_grompp_em.out'),
            stderr=join(output_path, name + '_grompp_em.err'),
        )

        em_energy = join(output_path, name + '_em_energy.edr')
        mdrun_fut = energy_minimization(
            em=em,
            em_energy=em_energy,
            inputs=[em_tpr_fut],
            outputs=[File(em_energy)],
            stdout=join(output_path, name + '_mdrun.out'),
            stderr=join(output_path, name + '_mdrun.err'),
        )

        # 6th step - energy analysis
        energy_result = join(output_path, name + '_potential.xvg')
        energy_selection = join(config_path, 'energy.selection')
        energy_analisis(
            em=em_energy,
            output=energy_result,
            selection=energy_selection,
            inputs=[mdrun_fut],
            outputs=[File(energy_result)],
            stdout=join(output_path, name + '_energy.out'),
            stderr=join(output_path, name + '_energy.err'),
        )

    parsl.wait_for_current_tasks()


if __name__ == "__main__":
    config_path = sys.argv[1]
    dataset_path = sys.argv[2]
    output_path = sys.argv[3]

    main(dataset_path, output_path, config_path)