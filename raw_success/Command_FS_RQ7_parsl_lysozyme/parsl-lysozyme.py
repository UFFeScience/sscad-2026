#!/usr/bin/env python3
"""
Tradução Parsl do fluxo de trabalho GROMACS do PyCOMPSs.
Executa uma série de comandos GROMACS em arquivos de proteína, usando tarefas Bash
para cada etapa e objetos File para transferência de dados.
"""

import os
import sys
from os.path import join

import parsl
from parsl.app.app import bash_app
from parsl.data_provider.files import File
from parsl.configs.local_threads import config

parsl.load(config)


# ------------------------------------------------------------
# Tarefas Bash (equivalentes às tarefas @binary originais)
# ------------------------------------------------------------
@bash_app
def generate_topology(protein, structure, topology, stdout=None, stderr=None):
    """gmx pdb2gmx -f protein.pdb -o structure.gro -p topology.top ..."""
    cmd = (f'gmx pdb2gmx -f {protein.filepath} -o {structure.filepath} '
           f'-p {topology.filepath} -ignh -ff amber03 -water spce')
    return cmd


@bash_app
def define_box(structure, structure_newbox, stdout=None, stderr=None):
    """gmx editconf -f structure.gro -o structure_newbox.gro ..."""
    cmd = (f'gmx editconf -f {structure.filepath} -o {structure_newbox.filepath} '
           f'-c -d 1.0 -bt cubic')
    return cmd


@bash_app
def add_solvate(structure_newbox, protein_solv, topology, stdout=None, stderr=None):
    """gmx solvate -cp structure_newbox.gro -cs spc216.gro -o protein_solv.gro ..."""
    cmd = (f'gmx solvate -cp {structure_newbox.filepath} '
           f'-cs spc216.gro -o {protein_solv.filepath} -p {topology.filepath}')
    return cmd


@bash_app
def assemble_tpr(conf, protein_solv, topology, output, stdout=None, stderr=None):
    """gmx grompp -f mdp_file -c protein_solv.gro -p topology.top -o output.tpr"""
    cmd = (f'gmx grompp -f {conf.filepath} -c {protein_solv.filepath} '
           f'-p {topology.filepath} -o {output.filepath}')
    return cmd


@bash_app
def replace_solvent_with_ions(ions, output, topology, group, stdout=None, stderr=None):
    """
    gmx genion -s ions.tpr -o solv_ions.gro -p topology.top ... < genion.group
    O arquivo de grupo é fornecido via stdin.
    """
    cmd = (f'gmx genion -s {ions.filepath} -o {output.filepath} '
           f'-p {topology.filepath} -pname NA -nname CL -neutral '
           f'< {group.filepath}')
    return cmd


@bash_app
def energy_minimization(em, em_energy, stdout=None, stderr=None):
    """gmx mdrun -v -s em.tpr -e em_energy.edr (usando MPI implícito via mdrun)"""
    cmd = f'gmx mdrun -v -s {em.filepath} -e {em_energy.filepath}'
    return cmd


@bash_app
def energy_analisis(em, output, selection, stdout=None, stderr=None):
    """gmx energy -f em.edr -o output.xvg < energy.selection"""
    cmd = (f'gmx energy -f {em.filepath} -o {output.filepath} '
           f'< {selection.filepath}')
    return cmd


# ------------------------------------------------------------
# Fluxo principal
# ------------------------------------------------------------
def main(dataset_path, output_path, config_path):
    """
    Processa todas as proteínas encontradas em `dataset_path`.
    `output_path` – diretório para arquivos intermediários e finais.
    `config_path` – diretório contendo arquivos de parâmetros (ions.mdp, etc.).
    """
    os.makedirs(output_path, exist_ok=True)

    # Lista arquivos PDB no diretório de dados
    protein_names = []
    protein_pdbs = []
    for f in os.listdir(dataset_path):
        if f.endswith('.pdb') and os.path.isfile(join(dataset_path, f)):
            protein_names.append(f.split('.')[0])
            protein_pdbs.append(join(dataset_path, f))

    for name, pdb_path in zip(protein_names, protein_pdbs):
        # 1. Gerar topologia
        protein_file = File(pdb_path)
        structure_file = File(join(output_path, f'{name}.gro'))
        topology_file = File(join(output_path, f'{name}.top'))
        generate_topology(protein=protein_file,
                          structure=structure_file,
                          topology=topology_file).result()

        # 2. Definir caixa
        structure_newbox_file = File(join(output_path, f'{name}_newbox.gro'))
        define_box(structure=structure_file,
                   structure_newbox=structure_newbox_file).result()

        # 3. Adicionar solvente
        protein_solv_file = File(join(output_path, f'{name}_solv.gro'))
        add_solvate(structure_newbox=structure_newbox_file,
                    protein_solv=protein_solv_file,
                    topology=topology_file).result()

        # 4. Montar TPR com íons
        ions_conf_file = File(join(config_path, 'ions.mdp'))
        ions_file = File(join(output_path, f'{name}_ions.tpr'))
        assemble_tpr(conf=ions_conf_file,
                     protein_solv=protein_solv_file,
                     topology=topology_file,
                     output=ions_file).result()

        protein_solv_ions_file = File(join(output_path, f'{name}_solv_ions.gro'))
        group_file = File(join(config_path, 'genion.group'))
        replace_solvent_with_ions(ions=ions_file,
                                  output=protein_solv_ions_file,
                                  topology=topology_file,
                                  group=group_file).result()

        # 5. Minimização de energia
        minim_conf_file = File(join(config_path, 'minim.mdp'))
        em_file = File(join(output_path, f'{name}_em.tpr'))
        assemble_tpr(conf=minim_conf_file,
                     protein_solv=protein_solv_ions_file,
                     topology=topology_file,
                     output=em_file).result()

        em_energy_file = File(join(output_path, f'{name}_em_energy.edr'))
        energy_minimization(em=em_file,
                            em_energy=em_energy_file).result()

        # 6. Análise de energia
        energy_result_file = File(join(output_path, f'{name}_potential.xvg'))
        energy_selection_file = File(join(config_path, 'energy.selection'))
        energy_analisis(em=em_energy_file,
                        output=energy_result_file,
                        selection=energy_selection_file).result()

        print(f"Processamento de {name} concluído.")


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Uso: python gromacs_parsl.py <config_path> <dataset_path> <output_path>')
        sys.exit(1)

    config_path = sys.argv[1]
    dataset_path = sys.argv[2]
    output_path = sys.argv[3]
    main(dataset_path, output_path, config_path)