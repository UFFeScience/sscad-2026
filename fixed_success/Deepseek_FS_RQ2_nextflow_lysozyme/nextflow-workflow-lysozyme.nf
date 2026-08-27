/*
 * pipeline input parameters
 */
params.algorithm = 'kmeans'
params.k = 3
params.outdir = 'results'

params.kmeans_data = "$projectDir/data/kmeans_data.txt"
params.matrix_a = "$projectDir/data/matrix_a.txt"
params.matrix_b = "$projectDir/data/matrix_b.txt"
params.pdb = "/Users/plwms/Documents/dissertacao/testes/automation/fixtures/lysozyme/dataset/1AKI.pdb"
params.mdp = "/Users/plwms/Documents/dissertacao/testes/automation/fixtures/lysozyme/config/minim.mdp"

/*
 * KMeans
 */


process LYSOZYME {
    tag "Lysozyme GROMACS"
    cpus 8
    publishDir params.outdir, mode: 'copy'

    input:
    path pdb
    path mdp

    output:
    path 'processed.gro'
    path 'topol.top'
    path 'posre.itp'
    path 'em.tpr'
    path 'em.gro'
    path 'em.log'

    script:
    """
    gmx pdb2gmx -f $pdb -o processed.gro -ff gromos53a6 -water spc -ignh
    gmx grompp -f $mdp -c processed.gro -p topol.top -o em.tpr -maxwarn 2
    mpirun -np $task.cpus gmx_mpi mdrun -deffnm em -v
    """
}


workflow LYSOZYME_WF {
    LYSOZYME(params.pdb, params.mdp)
}


workflow {
    LYSOZYME_WF()
}
