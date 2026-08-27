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


process KMEANS {
    container 'python:3.9-slim'
    input:
        path input_file
    output:
        path 'kmeans_result.txt'
    script:
        """
        python3 <<EOF
        import sys
        import numpy as np
        from sklearn.cluster import KMeans

        data = np.loadtxt('${input_file}', delimiter=',')
        kmeans = KMeans(n_clusters=3, random_state=0).fit(data)
        labels = kmeans.labels_
        np.savetxt('kmeans_result.txt', labels, fmt='%d')
        EOF
        """
}


workflow {
    KMEANS ( file(params.input) )
}
