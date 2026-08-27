/*
 * Parâmetros de entrada do pipeline
 */
params.matmul_a    = "$projectDir/A.txt"
params.matmul_b    = "$projectDir/B.txt"
params.outdir      = "results"

/*
 * PROCESSO: Matmul
 * Realiza a multiplicação de matrizes
 */
process MATMUL {
    tag "Matrix Multiplication"
    publishDir "${params.outdir}/matmul", mode: 'copy'

    input:
    path mat_a
    path mat_b

    output:
    path "result_matrix.txt"

    script:
    """
    # Exemplo utilizando um script python hipotético
    python3 run_matmul.py $mat_a $mat_b > result_matrix.txt
    """
}

workflow {
    log.info """
        MATMUL WORKFLOW PIPELINE
        ===================================
        Matmul A/B   : ${params.matmul_a} / ${params.matmul_b}
        Outdir       : ${params.outdir}
        ===================================
    """

    // Execução do Matmul
    MATMUL(params.matmul_a, params.matmul_b)
}
