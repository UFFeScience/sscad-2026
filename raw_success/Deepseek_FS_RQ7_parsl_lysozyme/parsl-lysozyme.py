import os
import sys
from os import listdir
from os.path import isfile, join

import parsl
from parsl.app.app import bash_app
from parsl.configs.local_threads import config
from parsl.data_provider.files import File

parsl.load(config)

GMX_DIR = os.environ.get("GMX_BIN", "gmx")

if GMX_DIR == "gmx":
    GMX = "gmx"
    GMX_MPI = "gmx_mpi"
elif os.path.basename(GMX_DIR) == "gmx":
    GMX = GMX_DIR
    GMX_MPI = os.path.join(os.path.dirname(GMX_DIR), "gmx_mpi")
else:
    GMX = os.path.join(GMX_DIR, "gmx")
    GMX_MPI = os.path.join(GMX_DIR, "gmx_mpi")


@bash_app
def generate_topology(inputs, outputs):
    protein = inputs[0].filepath
    structure = outputs[0].filepath
    topology = outputs[1].filepath
    return f"{GMX} pdb2gmx -f {protein} -o {structure} -p {topology} -ignh -ff oplsaa -water spce"


@bash_app
def define_box(inputs, outputs):
    structure = inputs[0].filepath
    out = outputs[0].filepath
    return f"{GMX} editconf -f {structure} -o {out} -c -d 1.0 -bt cubic"


@bash_app
def add_solvate(inputs, outputs):
    structure_newbox = inputs[0].filepath
    topology = inputs[1].filepath
    out = outputs[0].filepath
    return f"{GMX} solvate -cp {structure_newbox} -cs spc216.gro -o {out} -p {topology}"


@bash_app
def assemble_tpr(inputs, outputs):
    conf = inputs[0].filepath
    structure = inputs[1].filepath
    topology = inputs[2].filepath
    out = outputs[0].filepath
    return f"{GMX} grompp -f {conf} -c {structure} -p {topology} -o {out}"


@bash_app
def replace_solvent_with_ions(inputs, outputs):
    ions = inputs[0].filepath
    topology = inputs[1].filepath
    group = inputs[2].filepath
    out = outputs[0].filepath
    return f"{GMX} genion -s {ions} -o {out} -p {topology} -pname NA -nname CL -neutral < {group}"


@bash_app
def energy_minimization(inputs, outputs):
    em = inputs[0].filepath
    out = outputs[0].filepath
    return f"mpirun -np 4 {GMX_MPI} mdrun -v -ntomp 0 -s {em} -e {out}"


@bash_app
def energy_analisis(inputs, outputs):
    em = inputs[0].filepath
    selection = inputs[1].filepath
    out = outputs[0].filepath
    return f"{GMX} energy -f {em} -o {out} < {selection}"


def main(dataset_path, output_path, config_path):
    os.makedirs(output_path, exist_ok=True)

    protein_names = []
    protein_pdbs = []

    for f in listdir(dataset_path):
        if isfile(join(dataset_path, f)):
            protein_names.append(f.split(".")[0])
            protein_pdbs.append(join(dataset_path, f))

    final_futures = []

    for name, pdb in zip(protein_names, protein_pdbs):
        structure = join(output_path, name + ".gro")
        topology = join(output_path, name + ".top")

        gen = generate_topology(
            inputs=[File(pdb)],
            outputs=[File(structure), File(topology)],
        )

        structure_newbox = join(output_path, name + "_newbox.gro")
        box = define_box(
            inputs=[gen.outputs[0]],
            outputs=[File(structure_newbox)],
        )

        protein_solv = join(output_path, name + "_solv.gro")
        solv = add_solvate(
            inputs=[box.outputs[0], gen.outputs[1]],
            outputs=[File(protein_solv)],
        )

        ions_conf = join(config_path, "ions.mdp")
        ions = join(output_path, name + "_ions.tpr")
        tpr_ions = assemble_tpr(
            inputs=[File(ions_conf), solv.outputs[0], gen.outputs[1]],
            outputs=[File(ions)],
        )

        protein_solv_ions = join(output_path, name + "_solv_ions.gro")
        group = join(config_path, "genion.group")
        replace = replace_solvent_with_ions(
            inputs=[tpr_ions.outputs[0], gen.outputs[1], File(group)],
            outputs=[File(protein_solv_ions)],
        )

        minim_conf = join(config_path, "minim.mdp")
        em = join(output_path, name + "_em.tpr")
        tpr_em = assemble_tpr(
            inputs=[File(minim_conf), replace.outputs[0], gen.outputs[1]],
            outputs=[File(em)],
        )

        em_energy = join(output_path, name + "_em_energy.edr")
        md = energy_minimization(
            inputs=[tpr_em.outputs[0]],
            outputs=[File(em_energy)],
        )

        energy_result = join(output_path, name + "_potential.xvg")
        energy_selection = join(config_path, "energy.selection")
        analysis = energy_analisis(
            inputs=[md.outputs[0], File(energy_selection)],
            outputs=[File(energy_result)],
        )

        final_futures.append(analysis)

    for f in final_futures:
        f.result()

    parsl.clear()


if __name__ == "__main__":
    config_path = sys.argv[1]
    dataset_path = sys.argv[2]
    output_path = sys.argv[3]

    main(dataset_path, output_path, config_path)