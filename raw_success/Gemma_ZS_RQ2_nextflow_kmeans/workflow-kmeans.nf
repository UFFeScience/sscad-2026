nextflow.enable.dsl=2

// Parameters
params.kmeans_data = "data_kmeans.txt"
params.outdir = "results"

process KMEANS {
    tag "KMeans"
    publishDir "${params.outdir}/kmeans", mode: 'copy'

    input:
    path data_file

    output:
    path "clusters.txt"

    script:
    """
    python3 ${projectDir}/kmeans.py ${data_file} > clusters.txt
    """
}

workflow {
    // Input Channels
    ch_kmeans_data = Channel.fromPath(params.kmeans_data)

    // Execution
    KMEANS(ch_kmeans_data)
}
