/*
 * pipeline input parameters
 */
params.kmeans_data = "$projectDir/data/kmeans/data.txt"
params.matmul_a = "$projectDir/data/matmul/A.txt"
params.matmul_b = "$projectDir/data/matmul/B.txt"
params.lysozyme_pdb = "$projectDir/1AKI.pdb"
params.outdir = "results"

/*
 * Process: KMEANS
 * Calculates clusters using a Python script.
 */


process LYSODYME {
    cpus 1
    publishDir params.outdir, mode: 'copy'

    input:
    path pdb_file

    output:
    path 'lysozyme_traj.gro'
    path 'potential.xvg'

    script:
    """
    # 1. Generate topology using pdb2gmx
    gmx pdb2gmx -f $pdb_file -o lysozyme_processed.gro -p topol.top -water spce -ff oplsaa -ignh

    # 2. Prepare simulation using gmx (grompp)
    # Create a minimal md.mdp file inline
    cat > md.mdp << 'EOF'
    ; MDP file for Lysozyme simulation
    title               = Lysozyme MD simulation
    define              = -DBOX_MDP
    integrator          = steep
    nsteps              = 10000
    dt                  = 0.001
    cutoff-scheme       = Verlet
    vdw-type            = Cut-off
    ; ...
    EOF

    gmx grompp -f md.mdp -c lysozyme_processed.gro -p topol.top -o lysozyme_em.tpr -maxwarn 1

    # 3. Run simulation using gmx_mpi (as requested)
    # -ntomp uses the cpus allocated by Nextflow
    gmx_mpi mdrun -s lysozyme_em.tpr -o traj.trr -c traj.gro -e traj.edr -ntomp $task.cpus

    # Rename outputs to match declared output paths
    mv traj.gro lysozyme_traj.gro

    # Extract energy to .xvg format
    gmx energy -f traj.edr -o potential.xvg << EOF
    Potential
    0
    EOF
    """
}


workflow {
    lysozyme_ch = Channel.fromPath(params.lysozyme_pdb)
    LYSODYME(lysozyme_ch)
}
