nextflow.enable.dsl=2

params.input = "dataset.csv"
params.k = 4
params.outdir = "results"

process KMEANS {

    publishDir params.outdir, mode: 'copy'

    input:
    path dataset

    output:
    path "clusters.csv"

    script:
    """
    python3 ${projectDir}/kmeans.py \
           --input $dataset \
           --k ${params.k} \
           --output clusters.csv
    """
}

workflow {

    Channel
        .fromPath(params.input)
        .set { dataset_ch }

    KMEANS(dataset_ch)
}