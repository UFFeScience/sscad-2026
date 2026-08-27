#!/usr/bin/env python3

import os
import sys
from os import listdir
from os.path import isfile, join

import parsl
from parsl.app.app import bash_app
from parsl.data_provider.files import File
from parsl.configs.local_threads import config

GMX_BIN_PREFIX = (os.environ["GMX_BIN"] + "/") if os.environ.get("GMX_BIN") else ""

parsl.load(config)


# ############ #
# Step 1 tasks #
# ############ #

@bash_app
def generate_topology(
    inputs=[],
    outputs=[],
    forcefield="oplsaa",
    water="spce",
    stdout=None,
    stderr=None,
):
    protein = inputs[0].filepath
    structure = outputs[0].filepath
    topology = outputs[1].filepath

    return (
        f'{GMX_BIN_PREFIX}gmx pdb2gmx '
        f'-f "{protein}" '
        f'-o "{structure}" '
        f'-p "{topology}" '
        '-ignh '
        f'-ff {forcefield} '
        f'-water {water}'
    )


# ############ #
# Step 2 tasks #
# ############ #

@bash_app
def define_box(
    inputs=[],
    outputs=[],
    distance="1.0",
    boxtype="cubic",
    stdout=None,
    stderr=None,
):
    structure = inputs[0].filepath
    structure_newbox = outputs[0].filepath

    return (
        f'{GMX_BIN_PREFIX}gmx editconf '
        f'-f "{structure}" '
        f'-o "{structure_newbox}" '
        '-c '
        f'-d {distance} '
        f'-bt {boxtype}'
    )


# ############ #
# Step 3 tasks #
# ############ #

@bash_app
def add_solvate(
    inputs=[],
    outputs=[],
    configuration_solvent="spc216.gro",
    stdout=None,
    stderr=None,
):
    structure_newbox = inputs[0].filepath
    topology = inputs[1].filepath
    protein_solv = outputs[0].filepath

    return (
        f'{GMX_BIN_PREFIX}gmx solvate '
        f'-cp "{structure_newbox}" '
        f'-cs {configuration_solvent} '
        f'-o "{protein_solv}" '
        f'-p "{topology}"'
    )


# ############ #
# Step 4 tasks #
# ############ #

@bash_app
def assemble_tpr(
    inputs=[],
    outputs=[],
    stdout=None,
    stderr=None,
):
    conf = inputs[0].filepath
    protein_solv = inputs[1].filepath
    topology = inputs[2].filepath
    output = outputs[0].filepath

    return (
        f'{GMX_BIN_PREFIX}gmx grompp '
        f'-f "{conf}" '
        f'-c "{protein_solv}" '
        f'-p "{topology}" '
        f'-o "{output}"'
    )


@bash_app
def replace_solvent_with_ions(
    inputs=[],
    outputs=[],
    pname="NA",
    nname="CL",
    stdout=None,
    stderr=None,
):
    ions = inputs[0].filepath
    topology = inputs[1].filepath
    group = inputs[2].filepath
    output = outputs[0].filepath

    return (
        f'{GMX_BIN_PREFIX}gmx genion '
        f'-s "{ions}" '
        f'-o "{output}" '
        f'-p "{topology}" '
        f'-pname {pname} '
        f'-nname {nname} '
        '-neutral '
        f'< "{group}"'
    )


# ############ #
# Step 5 tasks #
# ############ #

@bash_app
def energy_minimization(
    inputs=[],
    outputs=[],
    nproc=4,
    ompthreads=0,
    stdout=None,
    stderr=None,
):
    em = inputs[0].filepath
    em_energy = outputs[0].filepath

    return (
        f'mpirun -np {nproc} gmx_mpi mdrun '
        '-v '
        f'-ntomp {ompthreads} '
        f'-s "{em}" '
        f'-e "{em_energy}"'
    )


# ############ #
# Step 6 tasks #
# ############ #

@bash_app
def energy_analysis(
    inputs=[],
    outputs=[],
    stdout=None,
    stderr=None,
):
    em_energy = inputs[0].filepath
    selection = inputs[1].filepath
    output = outputs[0].filepath

    return (
        f'{GMX_BIN_PREFIX}gmx energy '
        f'-f "{em_energy}" '
        f'-o "{output}" '
        f'< "{selection}"'
    )


# ############# #
# MAIN FUNCTION #
# ############# #

def main(dataset_path, output_path, config_path):
    print("Starting demo")

    os.makedirs(output_path, exist_ok=True)

    protein_names = []
    protein_pdbs = []

    for f in listdir(dataset_path):
        if isfile(join(dataset_path, f)):
            protein_names.append(f.split(".")[0])
            protein_pdbs.append(join(dataset_path, f))

    proteins = zip(protein_names, protein_pdbs)

    result_futures = []

    for name, pdb in proteins:
        # 1st step - Generate topology
        structure_file = File(join(output_path, name + ".gro"))
        topology_file = File(join(output_path, name + ".top"))

        step1 = generate_topology(
            inputs=[File(pdb)],
            outputs=[structure_file, topology_file],
            stdout=join(output_path, name + "_01_pdb2gmx.out"),
            stderr=join(output_path, name + "_01_pdb2gmx.err"),
        )

        # 2nd step - Define box
        structure_newbox_file = File(join(output_path, name + "_newbox.gro"))

        step2 = define_box(
            inputs=[step1.outputs[0]],
            outputs=[structure_newbox_file],
            stdout=join(output_path, name + "_02_editconf.out"),
            stderr=join(output_path, name + "_02_editconf.err"),
        )

        # 3rd step - Add solvate
        protein_solv_file = File(join(output_path, name + "_solv.gro"))

        step3 = add_solvate(
            inputs=[step2.outputs[0], step1.outputs[1]],
            outputs=[protein_solv_file],
            stdout=join(output_path, name + "_03_solvate.out"),
            stderr=join(output_path, name + "_03_solvate.err"),
        )

        # 4th step - Add ions
        ions_conf_file = File(join(config_path, "ions.mdp"))
        ions_file = File(join(output_path, name + "_ions.tpr"))

        step4a = assemble_tpr(
            inputs=[ions_conf_file, step3.outputs[0], step1.outputs[1]],
            outputs=[ions_file],
            stdout=join(output_path, name + "_04a_grompp_ions.out"),
            stderr=join(output_path, name + "_04a_grompp_ions.err"),
        )

        protein_solv_ions_file = File(
            join(output_path, name + "_solv_ions.gro")
        )

        group_file = File(join(config_path, "genion.group"))

        step4b = replace_solvent_with_ions(
            inputs=[step4a.outputs[0], step1.outputs[1], group_file],
            outputs=[protein_solv_ions_file],
            stdout=join(output_path, name + "_04b_genion.out"),
            stderr=join(output_path, name + "_04b_genion.err"),
        )

        # 5th step - Energy minimization
        minim_conf_file = File(join(config_path, "minim.mdp"))
        em_file = File(join(output_path, name + "_em.tpr"))

        step5a = assemble_tpr(
            inputs=[minim_conf_file, step4b.outputs[0], step1.outputs[1]],
            outputs=[em_file],
            stdout=join(output_path, name + "_05a_grompp_em.out"),
            stderr=join(output_path, name + "_05a_grompp_em.err"),
        )

        em_energy_file = File(join(output_path, name + "_em_energy.edr"))

        step5b = energy_minimization(
            inputs=[step5a.outputs[0]],
            outputs=[em_energy_file],
            nproc=4,
            stdout=join(output_path, name + "_05b_mdrun.out"),
            stderr=join(output_path, name + "_05b_mdrun.err"),
        )

        # 6th step - Energy analysis
        energy_result_file = File(join(output_path, name + "_potential.xvg"))
        energy_selection_file = File(join(config_path, "energy.selection"))

        step6 = energy_analysis(
            inputs=[step5b.outputs[0], energy_selection_file],
            outputs=[energy_result_file],
            stdout=join(output_path, name + "_06_energy.out"),
            stderr=join(output_path, name + "_06_energy.err"),
        )

        result_futures.append(step6)

    # Equivalent to a final compss_barrier
    for fut in result_futures:
        fut.result()

    print("Workflow finished")

    for fut in result_futures:
        print("Generated:", fut.outputs[0].filepath)


if __name__ == "__main__":
    config_path = sys.argv[1]
    dataset_path = sys.argv[2]
    output_path = sys.argv[3]

    main(dataset_path, output_path, config_path)
    parsl.clear()