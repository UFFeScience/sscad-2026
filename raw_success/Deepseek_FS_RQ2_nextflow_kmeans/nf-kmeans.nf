#!/usr/bin/env nextflow

/*
 * pipeline input parameters
 */
params.data_file = "$projectDir/dataset.csv"
params.k = 3
params.max_iter = 100
params.outdir = "results"

/*
 * KMEANS process - Executa o algoritmo KMeans
 */
process KMEANS {
    publishDir params.outdir, mode: 'copy'
    
    input:
    path data_file
    val k
    val max_iter
    
    output:
    path "kmeans_results.txt"
    path "kmeans_clusters.csv"
    
    script:
    """
    python3 <<EOF
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import sys

# Ler dados
data = pd.read_csv('${data_file}')
X = data.values

# Executar KMeans
kmeans = KMeans(n_clusters=${k}, max_iter=${max_iter}, random_state=42)
labels = kmeans.fit_predict(X)

# Salvar resultados
with open('kmeans_results.txt', 'w') as f:
    f.write(f'KMeans Results\\n')
    f.write(f'==============\\n')
    f.write(f'Number of clusters: ${k}\\n')
    f.write(f'Maximum iterations: ${max_iter}\\n')
    f.write(f'Inertia: {kmeans.inertia_:.4f}\\n')
    f.write(f'Number of iterations: {kmeans.n_iter_}\\n')

# Salvar clusters
data['cluster'] = labels
data.to_csv('kmeans_clusters.csv', index=False)
EOF
    """
}

workflow {
    log.info """\
        K M E A N S   P I P E L I N E
        =============================
        data file    : ${params.data_file}
        k clusters   : ${params.k}
        max iter     : ${params.max_iter}
        outdir       : ${params.outdir}
        """.stripIndent(true)
    
    KMEANS(
        file(params.data_file),
        params.k,
        params.max_iter
    )
}