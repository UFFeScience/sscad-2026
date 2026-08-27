#!/usr/bin/env nextflow
// ============================================================================
// Script Nextflow para executar três algoritmos separadamente:
// KMeans, MatMul e Lysozyme (GROMACS).
// Cada processo pode ser executado independentemente, fornecendo os parâmetros
// de entrada correspondentes.
// ============================================================================

nextflow.enable.dsl=2

// Parâmetros de entrada (arquivos). Se não fornecidos, o processo correspondente
// não será executado.
params.kmeans_input = null
params.matmul_input = null
params.lysozyme_pdb = "/Users/plwms/Documents/dissertacao/testes/automation/fixtures/lysozyme/dataset/1AKI.pdb"
params.lysozyme_ff  = 'gromos53a6'   // força campo para pdb2gmx
params.lysozyme_water = 'spc'        // modelo de água

// ----------------------------------------------------------------------------
// Processo: KMeans
// Executa um script Python (kmeans.py) que recebe um arquivo de entrada e
// produz um resultado (ex: clusters.txt).
// ----------------------------------------------------------------------------


process LYSOZYME {
    input:
    path pdb_file

    output:
    path "em.gro", emit: minimized_structure
    path "topol.top", emit: topology

    script:
    // Definir nomes de arquivos intermediários
    def processed = "processed.gro"
    def box = "box.gro"
    def solvated = "solvated.gro"
    def ions = "ions.gro"
    def em_tpr = "em.tpr"
    def em_gro = "em.gro"

    """
    # 1. Gerar topologia a partir do PDB
    gmx pdb2gmx -f ${pdb_file} -o ${processed} -ff ${params.lysozyme_ff} -water ${params.lysozyme_water} -ignh

    # 2. Definir caixa cúbica
    gmx editconf -f ${processed} -o ${box} -c -d 1.0 -bt cubic

    # 3. Solvatação
    gmx solvate -cp ${box} -cs spc216.gro -o ${solvated} -p topol.top

    # 4. Preparar íons (neutralizar)
    gmx grompp -f /Users/plwms/Documents/dissertacao/testes/automation/fixtures/lysozyme/config/ions.mdp -c ${solvated} -p topol.top -o ions.tpr -maxwarn 2
    echo "SOL" | gmx genion -s ions.tpr -o ${ions} -p topol.top -pname NA -nname CL -neutral

    # 5. Minimização de energia
    gmx grompp -f /Users/plwms/Documents/dissertacao/testes/automation/fixtures/lysozyme/config/minim.mdp -c ${ions} -p topol.top -o ${em_tpr} -maxwarn 2
    gmx_mpi mdrun -deffnm em -v

    # O arquivo em.gro e topol.top serão emitidos como saída
    """
}


workflow {
    LYSOZYME( file(params.lysozyme_pdb) )
}
