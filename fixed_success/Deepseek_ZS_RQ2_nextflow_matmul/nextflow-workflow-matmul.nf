#!/usr/bin/env nextflow

/*
 * Script Nextflow para executar workflows de KMeans, Matmul e Lysozyme.
 * Uso:
 *   nextflow run main.nf --algo kmeans   [--input data.csv]
 *   nextflow run main.nf --algo matmul   [--input matrixA.txt]
 *   nextflow run main.nf --algo lysozyme [--input protein.pdb]
 */

params.algo  = 'kmeans'          // escolha: kmeans, matmul, lysozyme
params.input = '/Users/plwms/Documents/dissertacao/testes/RQ2/Deepseek/C-EXEMPLOS/A.csv'       // arquivo de entrada padrão

// ----------------------------------------------------------------------------
// PROCESSOS
// ----------------------------------------------------------------------------

// Processo para KMeans (usando Python)


process MATMUL {
    container 'python:3.9-slim'
    input:
        path input_file
    output:
        path 'matmul_result.txt'
    script:
        """
        python3 <<EOF
        import numpy as np

        A = np.loadtxt('${input_file}', delimiter=',')
        B = np.random.rand(A.shape[1], A.shape[1])
        C = np.dot(A, B)
        np.savetxt('matmul_result.txt', C, delimiter=',')
        EOF
        """
}


workflow {
    MATMUL ( file(params.input) )
}
