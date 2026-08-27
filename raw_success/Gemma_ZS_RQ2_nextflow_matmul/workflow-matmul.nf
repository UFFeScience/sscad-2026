nextflow.enable.dsl=2

// Parameters
params.mat1 = "mat1.txt"
params.mat2 = "mat2.txt"
params.outdir = "results"

process MATMUL {
    tag "MatMul"
    publishDir "${params.outdir}/matmul", mode: 'copy'

    input:
    path m1
    path m2

    output:
    path "result_matrix.txt"

    script:
    """
    python3 ${projectDir}/matmul.py ${m1} ${m2} > result_matrix.txt
    """
}

workflow {
    // Input Channels
    ch_mat1 = Channel.fromPath(params.mat1)
    ch_mat2 = Channel.fromPath(params.mat2)

    // Execution
    MATMUL(ch_mat1, ch_mat2)
}
