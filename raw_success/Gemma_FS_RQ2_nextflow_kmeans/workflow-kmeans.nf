/*
 * Parâmetros de entrada do pipeline
 */
params.kmeans_data = "$projectDir/dataset.csv"
params.outdir      = "results"

/*
 * PROCESSO: KMeans
 * Assume-se a existência de um script python ou comando de CLI
 */
process KMEANS {
    tag "KMeans Clustering"
    publishDir "${params.outdir}/kmeans", mode: 'copy'

    input:
    path data_file

    output:
    path "clusters_result.txt"

    script:
    """
    # Exemplo utilizando um script python hipotético
    python3 run_kmeans.py --input $data_file --output clusters_result.txt
    """
}

workflow {
    log.info """
        KMEANS WORKFLOW PIPELINE
        ===================================
        KMeans Data  : ${params.kmeans_data}
        Outdir       : ${params.outdir}
        ===================================
    """

    // Execução do KMeans
    KMEANS(params.kmeans_data)
}
