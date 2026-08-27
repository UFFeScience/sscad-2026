nextflow.enable.dsl=2

params.matrixA="A.csv"
params.matrixB="B.csv"
params.outdir="results"

process MATMUL {

    publishDir params.outdir, mode:'copy'

    input:
    path A
    path B

    output:
    path "results.csv"

    script:
    """
    python ${projectDir}/matmul.py \
           --A $A \
           --B $B \
           --output results.csv
    """
}

workflow {

    Channel.fromPath(params.matrixA).set { A_ch }

    Channel.fromPath(params.matrixB).set { B_ch }

    MATMUL(A_ch,B_ch)
}