import os
import subprocess
import numpy as np
import argparse
from pycompss.api.task import task
from pycompss.api.api import compss_wait_on

# ==============================================================================
# WORKFLOW: LYSOZYME (GROMACS)
# Estratégia: Encapsular comandos de shell do GROMACS em tasks.
# Assume-se que gmx e gmx_mpi estão no PATH dos workers.
# ==============================================================================

@task
def gmx_pdb2gmx(pdb_file, output_prefix):
    """Gera a topologia utilizando pdb2gmx."""
    # Exemplo de comando: gmx pdb2gmx -f protein.pdb -o processed.gro -p topol.top
    # O comando 'echo "1"' simula a escolha da força de campo (Force Field)
    cmd = f'echo "1" | gmx pdb2gmx -f {pdb_file} -o {output_prefix}.gro -p {output_prefix}.top'
    subprocess.run(cmd, shell=True, check=True)
    return f"{output_prefix}.gro"

@task
def gmx_editconf(gro_file, output_prefix):
    """Define a caixa de simulação."""
    cmd = f'gmx editconf -f {gro_file} -o {output_prefix}_box.gro -c -d 1.0 -bt cubic'
    subprocess.run(cmd, shell=True, check=True)
    return f"{output_prefix}_box.gro"

@task
def gmx_solvate(gro_file, top_file, output_prefix):
    """Adiciona moléculas de água."""
    cmd = f'gmx solvate -cp {gro_file} -cs spc216.gro -o {output_prefix}_solv.gro -p {top_file}'
    subprocess.run(cmd, shell=True, check=True)
    return f"{output_prefix}_solv.gro"

def run_lysozyme(pdb_input):
    print("Iniciando Workflow Lysozyme...")
    prefix = "lysozyme_sim"

    # Fluxo de dependências: pdb2gmx -> editconf -> solvate
    gro_topology = gmx_pdb2gmx(pdb_input, prefix)
    gro_box = gmx_editconf(gro_topology, prefix)
    # Para o solvate, precisamos do arquivo de topologia gerado no primeiro passo
    top_file = f"{prefix}.top"
    gro_solvated = gmx_solvate(gro_box, top_file, prefix)

    final_file = compss_wait_on(gro_solvated)
    print(f"Workflow Lysozyme concluído. Arquivo final: {final_file}")
    return final_file

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='PyCOMPSs Lysozyme Workflow')
    parser.add_argument('-f', '--file', type=str, required=True, help='Arquivo PDB para Lysozyme')
    args = parser.parse_args()

    run_lysozyme(args.file)

if __name__ == "__main__":
    main()
