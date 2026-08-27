nextflow.enable.dsl=2


/*
 * Workflow Matmul
 * Executa a multiplicação de matrizes como processo independente
 */


process RUN_MATMUL {

    tag "Matrix Multiplication"

    input:
    path script
    val size


    output:
    path "matmul_result.txt"


    script:

    """
    python3 ${script} ${size}
    """
}



workflow {


    /*
     * Parâmetros do experimento
     */


    matrix_size = Channel.of(1000)



    /*
     * Execução do workflow
     */


    RUN_MATMUL(file("${projectDir}/nextflow-matmul.py"), matrix_size)


}
