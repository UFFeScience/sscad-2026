/*
 * Parâmetros de entrada do pipeline
 */
params.lysozyme_pdb = "$projectDir/1aki.pdb"
params.outdir      = "results"

/*
 * PROCESSO: Lysozyme
 * Utiliza GROMACS para gerar a topologia (pdb2gmx)
 */
process LYSOZYME {
    tag "GROMACS Lysozyme Topology"
    publishDir "${params.outdir}/lysozyme", mode: 'copy'

    // Define o uso de MPI se disponível, caso contrário usa o gmx padrão
    cpus 4

    input:
    path pdb_file

    output:
    path "topol.top", emit: topology
    path "processed.gro", emit: structure

    script:
    """
    # O comando pdb2gmx geralmente pede interação do usuário para escolher a força (forcefield).
    # Usamos o 'echo' para automatizar a escolha (ex: 15 para AMBER)
    echo "15" | gmx pdb2gmx -f $pdb_file -o processed.gro -p topol.top
    """
}

workflow {
    log.info """
        LYSOZYME WORKFLOW PIPELINE
        ===================================
        Lysozyme PDB : ${params.lysozyme_pdb}
        Outdir       : ${params.outdir}
        ===================================
    """

    // Execução do Lysozyme
    LYSOZYME(params.lysozyme_pdb)
}
