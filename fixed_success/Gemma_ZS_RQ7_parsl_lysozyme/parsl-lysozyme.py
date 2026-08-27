import sys
import os
from os import listdir
from os.path import isfile, join
import parsl
from parsl.app.app import bash_app, python_app
from parsl.configs.local_threads import config

parsl.load(config)

# Definindo o binário GMX como variável de ambiente ou caminho fixo
GMX_BIN = os.environ.get('GMX_BIN', '/usr/local/gromacs/bin')

@bash_app
def generate_topology(protein, structure, topology):
    return f"gmx pdb2gmx -f {protein} -o {structure} -p {topology} -ignh -ff oplsaa -water spce"

@bash_app
def define_box(structure, structure_newbox):
    return f"gmx editconf -f {structure} -o {structure_newbox} -c -d 1.0 -bt cubic"

@bash_app
def add_solvate(structure_newbox, protein_solv, topology):
    return f"gmx solvate -cp {structure_newbox} -cs spc216.gro -o {protein_solv} -p {topology}"

@bash_app
def assemble_tpr(conf, protein_solv, topology, output):
    return f"gmx grompp -f {conf} -c {protein_solv} -p {topology} -o {output}"

@bash_app
def replace_solvent_with_ions(ions, output, topology, group):
    return f"gmx genion -s {ions} -o {output} -p {topology} -pname NA -nname CL -neutral < {group}"

@bash_app
def energy_minimization(em, em_energy):
    # Simulando mpirun via shell_app
    return f"mpirun gmx_mpi mdrun -v -s {em} -e {em_energy}"

@bash_app
def energy_analisis(em, output, selection):
    return f"gmx energy -f {em} -o {output} < {selection}"

def main(dataset_path, output_path, config_path):
    print("Starting demo")
    
    proteins = []
    for f in listdir(dataset_path):
        if isfile(join(dataset_path, f)):
            proteins.append((f.split('.')[0], join(dataset_path, f)))

    for name, pdb in proteins:
        # Definição de caminhos
        structure = join(output_path, name + '.gro')
        topology = join(output_path, name + '.top')
        structure_newbox = join(output_path, name + '_newbox.gro')
        protein_solv = join(output_path, name + '_solv.gro')
        ions_conf = join(config_path, 'ions.mdp')
        ions = join(output_path, name + '_ions.tpr')
        protein_solv_ions = join(output_path, name + '_solv_ions.gro')
        group = join(config_path, 'genion.group')
        minim_conf = join(config_path, 'minim.mdp')
        em = join(output_path, name + '_em.tpr')
        em_energy = join(output_path, name + '_em_energy.edr')
        energy_result = join(output_path, name + '_potential.xvg')
        energy_selection = join(config_path, 'energy.selection')

        # Workflow: No Parsl, encadeamos os futures para garantir a ordem
        f1 = generate_topology(pdb, structure, topology)
        f1.result()  # precisa terminar antes: define_box le o .gro que generate_topology escreve
        f2 = define_box(structure, structure_newbox)
        f2.result()
        f3 = add_solvate(structure_newbox, protein_solv, topology).result()
        f4 = assemble_tpr(ions_conf, protein_solv, topology, ions).result()
        f5 = replace_solvent_with_ions(ions, protein_solv_ions, topology, group).result()
        f6 = assemble_tpr(minim_conf, protein_solv_ions, topology, em).result()
        f7 = energy_minimization(em, em_energy).result()
        f8 = energy_analisis(em_energy, energy_result, energy_selection).result()

if __name__=='__main__':
    if len(sys.argv) < 4:
        print("Usage: script.py <config_path> <dataset_path> <output_path>")
        sys.exit(1)
    main(sys.argv[2], sys.argv[3], sys.argv[1])