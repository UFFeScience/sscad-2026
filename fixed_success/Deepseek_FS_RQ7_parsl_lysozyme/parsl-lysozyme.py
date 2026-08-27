#!/usr/bin/env python3
import os
import sys
from parsl.app.app import bash_app
from parsl.data_provider.files import File
import parsl
from parsl.configs.local_threads import config

parsl.load(config)

# ############ #
# Step 1 tasks #
# ############ #
@bash_app
def generate_topology(protein, structure, topology, stdout='gmx_pdb2gmx.out', stderr='gmx_pdb2gmx.err'):
    return f"gmx pdb2gmx -f {protein} -o {structure} -p {topology} -ignh -ff oplsaa -water spce"

# ############ #
# Step 2 tasks #
# ############ #
@bash_app
def define_box(structure, structure_newbox, stdout='gmx_editconf.out', stderr='gmx_editconf.err'):
    return f"gmx editconf -f {structure} -o {structure_newbox} -c -d 1.0 -bt cubic"

# ############ #
# Step 3 tasks #
# ############ #
@bash_app
def add_solvate(structure_newbox, topology, protein_solv, stdout='gmx_solvate.out', stderr='gmx_solvate.err'):
    return f"gmx solvate -cp {structure_newbox} -cs spc216.gro -o {protein_solv} -p {topology}"

# ############ #
# Step 4 tasks #
# ############ #
@bash_app
def assemble_tpr(conf, protein_solv, topology, output, stdout='gmx_grompp.out', stderr='gmx_grompp.err'):
    return f"gmx grompp -f {conf} -c {protein_solv} -p {topology} -o {output} -maxwarn 2"

@bash_app
def replace_solvent_with_ions(ions, output, topology, group, stdout='gmx_genion.out', stderr='gmx_genion.err'):
    # O arquivo 'group' será fornecido via stdin (redirecionamento)
    return f"gmx genion -s {ions} -o {output} -p {topology} -pname NA -nname CL -neutral < {group}"

# ############ #
# Step 5 tasks #
# ############ #
@bash_app
def energy_minimization(em, em_energy, stdout='gmx_mdrun.out', stderr='gmx_mdrun.err'):
    # Para MPI, poderíamos usar mpirun, mas aqui usamos gmx_mpi com -ntomp 0
    return f"gmx_mpi mdrun -v -s {em} -e {em_energy} -ntomp 0"

# ############ #
# Step 6 tasks #
# ############ #
@bash_app
def energy_analisis(em, output, selection, stdout='gmx_energy.out', stderr='gmx_energy.err'):
    return f"gmx energy -f {em} -o {output} < {selection}"

# ############# #
# MAIN FUNCTION #
# ############# #
def main(dataset_path, output_path, config_path):
    print("Starting demo")

    from os import listdir
    from os.path import isfile, join

    protein_names = []
    protein_pdbs = []
    for f in listdir(dataset_path):
        if isfile(join(dataset_path, f)):
            protein_names.append(f.split('.')[0])
            protein_pdbs.append(join(dataset_path, f))
    proteins = zip(protein_names, protein_pdbs)

    result_image_paths = []
    for name, pdb in proteins:
        # Step 1
        structure = join(output_path, name + '.gro')
        topology = join(output_path, name + '.top')
        generate_topology(protein=pdb, structure=structure, topology=topology).result()

        # Step 2
        structure_newbox = join(output_path, name + '_newbox.gro')
        define_box(structure=structure, structure_newbox=structure_newbox).result()

        # Step 3
        protein_solv = join(output_path, name + '_solv.gro')
        add_solvate(structure_newbox=structure_newbox, topology=topology,
                    protein_solv=protein_solv).result()

        # Step 4
        ions_conf = join(config_path, 'ions.mdp')
        ions = join(output_path, name + '_ions.tpr')
        assemble_tpr(conf=ions_conf, protein_solv=protein_solv,
                     topology=topology, output=ions).result()

        protein_solv_ions = join(output_path, name + '_solv_ions.gro')
        group = join(config_path, 'genion.group')
        replace_solvent_with_ions(ions=ions, output=protein_solv_ions,
                                  topology=topology, group=group).result()

        # Step 5
        minim_conf = join(config_path, 'minim.mdp')
        em = join(output_path, name + '_em.tpr')
        assemble_tpr(conf=minim_conf, protein_solv=protein_solv_ions,
                     topology=topology, output=em).result()

        em_energy = join(output_path, name + '_em_energy.edr')
        energy_minimization(em=em, em_energy=em_energy).result()

        # Step 6
        energy_result = join(output_path, name + '_potential.xvg')
        energy_selection = join(config_path, 'energy.selection')
        energy_analisis(em=em_energy, output=energy_result,
                        selection=energy_selection).result()

if __name__ == '__main__':
    config_path = sys.argv[1]
    dataset_path = sys.argv[2]
    output_path = sys.argv[3]
    main(dataset_path, output_path, config_path)
    parsl.clear()