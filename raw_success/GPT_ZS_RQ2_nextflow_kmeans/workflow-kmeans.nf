nextflow.enable.dsl=2


/*
 * Workflow KMeans
 * Executa o KMeans como processo independente
 */


process RUN_KMEANS {

    tag "KMeans"

    input:
    path script
    val size

    output:
    path "kmeans_result.txt"


    script:

    """
    python3 ${script} ${size}
    """
}



workflow {


    /*
     * Parâmetros do experimento
     */


    kmeans_size = Channel.of(10000)



    /*
     * Execução do workflow
     */


    RUN_KMEANS(file("${projectDir}/nextflow-kmeans.py"), kmeans_size)


}
