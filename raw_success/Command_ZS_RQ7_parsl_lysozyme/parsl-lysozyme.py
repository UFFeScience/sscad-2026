import subprocess
import os
import sys

def run_command(cmd, description=""):
    """
    Helper to run shell commands.
    """
    print(f"Running: {description}")
    try:
        subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"Error in {description}: {e.stderr.decode()}")
        raise

def generate_topology(protein, structure, topology):
    cmd = f"gmx pdb2gmx -f {protein} -o {structure} -p {topology} -ignh -ff amber03 -water spce"
    run_command(cmd, "Step 1: Generate Topology")

def define_box(structure, structure_newbox):
    cmd = f"gmx editconf -f {structure} -o {structure_newbox} -c -d 1.0 -bt cubic"
    run_command(cmd, "Step 2: Define Box")

def add_solvate(structure_newbox, protein_solv, topology):
    cmd = f"gmx solvate -cp {structure_newbox} -cs spc216.gro -o {protein_solv} -p {topology}"
    run_command(cmd, "Step 3: Add Solvate")

def assemble_tpr(conf, protein_solv, topology, output):
    cmd = f"gmx grompp -f {conf} -c {protein_solv} -p {topology} -o {output}"
    run_command(cmd, "Step 4: Assemble TPR")

def replace_solvent_with_ions(ions, output, topology, group):
    cmd = f"gmx genion -s {ions} -o {output} -p {topology} -pname NA -nname CL -neutral < {group}"
    run_command(cmd, "Step 4b: Replace Solvent with Ions")

def energy_minimization(em, em_energy):
    # Note: -ntomp is set to 0 as per original, though mdrun might default differently
    cmd = f"gmx mdrun -v -s {em} -e {em_energy} -ntomp 0"
    run_command(cmd, "Step 5: Energy Minimization")

def energy_analisis(em, output, selection):
    cmd = f"gmx energy -f {em} -o {output} < {selection}"
    run_command(cmd, "Step 6: Energy Analysis")

def main(dataset_path, output_path, config_path):
    print("Starting demo")
    protein_names = []
    protein_pdbs = []

    for f in os.listdir(dataset_path):
        if os.path.isfile(os.path.join(dataset_path, f)):
            protein_names.append(f.split('.')[0])
            protein_pdbs.append(os.path.join(dataset_path, f))
            
    for name, pdb in zip(protein_names, protein_pdbs):
        structure = os.path.join(output_path, f"{name}.gro")
        topology = os.path.join(output_path, f"{name}.top")
        generate_topology(pdb, structure, topology)
        
        structure_newbox = os.path.join(output_path, f"{name}_newbox.gro")
        define_box(structure, structure_newbox)
        
        protein_solv = os.path.join(output_path, f"{name}_solv.gro")
        add_solvate(structure_newbox, protein_solv, topology)
        
        ions_conf = os.path.join(config_path, 'ions.mdp')
        ions = os.path.join(output_path, f"{name}_ions.tpr")
        assemble_tpr(ions_conf, protein_solv, topology, ions)
        
        protein_solv_ions = os.path.join(output_path, f"{name}_solv_ions.gro")
        group = os.path.join(config_path, 'genion.group')
        replace_solvent_with_ions(ions, protein_solv_ions, topology, group)
        
        minim_conf = os.path.join(config_path, 'minim.mdp')
        em = os.path.join(output_path, f"{name}_em.tpr")
        assemble_tpr(minim_conf, protein_solv_ions, topology, em)
        
        em_energy = os.path.join(output_path, f"{name}_em_energy.edr")
        energy_minimization(em, em_energy)
        
        energy_result = os.path.join(output_path, f"{name}_potential.xvg")
        energy_selection = os.path.join(config_path, 'energy.selection')
        energy_analisis(em_energy, energy_result, energy_selection)

if __name__ == '__main__':
    # Usage: python script.py <config_path> <dataset_path> <output_path>
    # Adjusted order based on original main signature vs usage in script
    # Original main signature: main(dataset_path, output_path, config_path)
    # Argv in original script: sys.argv[1]=config, [2]=dataset, [3]=output
    # We will follow the original script's execution flow:
    
    if len(sys.argv) != 4:
        print("Usage: python script.py <config_path> <dataset_path> <output_path>")
        sys.exit(1)
        
    config_path = sys.argv[1]
    dataset_path = sys.argv[2]
    output_path = sys.argv[3]
    
    main(dataset_path, output_path, config_path)