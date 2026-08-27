import os
import time
import argparse
import numpy as np
from pprint import pprint
from pycompss.api.task import task
from pycompss.api.api import compss_wait_on
from pycompss.api.parameter import FILE_IN, FILE_OUT

# ============================================================================
# LYSOZYME IMPLEMENTATION (GROMACS)
# ============================================================================

@task(returns=str)
def download_lysozyme():
    """Download lysozyme PDB file if not exists."""
    import urllib.request

    pdb_file = "lysozyme.pdb"
    if not os.path.exists(pdb_file):
        url = "https://files.rcsb.org/download/1AKI.pdb"
        urllib.request.urlretrieve(url, pdb_file)
        return "Downloaded lysozyme PDB file"
    return "Lysozyme PDB file already exists"

@task(returns=str)
def prepare_topology(pdb_file, forcefield="charmm36-jul2022"):
    """Prepare GROMACS topology using pdb2gmx."""
    import subprocess

    # Use pdb2gmx to generate topology
    cmd = f"gmx pdb2gmx -f {pdb_file} -o processed.gro -p topol.top -ignh -ff {forcefield} -water spc"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        return f"Error preparing topology: {result.stderr}"
    return "Topology prepared successfully"

@task(returns=str)
def solvate_system():
    """Solvate the system."""
    import subprocess

    # Define box
    cmd1 = "gmx editconf -f processed.gro -o box.gro -c -d 1.0 -bt cubic"
    subprocess.run(cmd1, shell=True, check=True)

    # Solvate
    cmd2 = "gmx solvate -cp box.gro -cs spc216.gro -o solvated.gro -p topol.top"
    result = subprocess.run(cmd2, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        return f"Error solvating: {result.stderr}"
    return "System solvated successfully"

@task(returns=str)
def add_ions():
    """Add ions to neutralize the system."""
    import subprocess

    # Prepare for ions
    cmd1 = "gmx grompp -f ions.mdp -c solvated.gro -p topol.top -o ions.tpr"
    subprocess.run(cmd1, shell=True, check=True)

    # Add ions
    cmd2 = "echo SOL | gmx genion -s ions.tpr -o solvated_ions.gro -p topol.top -pname NA -nname CL -neutral"
    result = subprocess.run(cmd2, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        return f"Error adding ions: {result.stderr}"
    return "Ions added successfully"

@task(returns=str)
def energy_minimization():
    """Perform energy minimization."""
    import subprocess

    # Prepare for minimization
    cmd1 = "gmx grompp -f minim.mdp -c solvated_ions.gro -p topol.top -o em.tpr"
    subprocess.run(cmd1, shell=True, check=True)

    # Run minimization
    cmd2 = "gmx mdrun -v -deffnm em"
    result = subprocess.run(cmd2, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        return f"Error during minimization: {result.stderr}"
    return "Energy minimization completed successfully"

def create_mdp_files():
    """Create necessary MDP files for GROMACS."""
    with open("ions.mdp", "w") as f:
        f.write("""integrator = steep
nsteps = 5000
nstlist = 1
ns_type = grid
coulombtype = PME
rcoulomb = 1.0
rvdw = 1.0
pbc = xyz
""")

    with open("minim.mdp", "w") as f:
        f.write("""integrator = steep
nsteps = 5000
nstlist = 1
ns_type = grid
coulombtype = PME
rcoulomb = 1.0
rvdw = 1.0
pbc = xyz
emtol = 10.0
emstep = 0.01
""")

@task(returns=dict)
def run_lysozyme_simulation():
    """Run the complete lysozyme simulation workflow."""
    results = {}

    # Create MDP files
    create_mdp_files()

    # Download structure
    results['download'] = download_lysozyme()

    # Prepare topology
    results['topology'] = prepare_topology("lysozyme.pdb")

    # Solvate system
    results['solvation'] = solvate_system()

    # Add ions
    results['ions'] = add_ions()

    # Energy minimization
    results['minimization'] = energy_minimization()

    return results

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function to execute the Lysozyme simulation."""
    start_time = time.time()

    print("Running Lysozyme simulation...")

    # Check for GROMACS
    import shutil
    if not shutil.which('gmx') and not shutil.which('gmx_mpi'):
        print("ERROR: GROMACS (gmx or gmx_mpi) not found in PATH")
        print("Please install GROMACS and ensure it's available in your PATH")
        return

    results = run_lysozyme_simulation()
    results = compss_wait_on(results)

    print("Lysozyme Simulation Results:")
    for step, message in results.items():
        print(f"  {step}: {message}")

    total_time = time.time() - start_time
    print(f"Total execution time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()
