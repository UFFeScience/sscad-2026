nextflow.enable.dsl=2

// Parâmetros de entrada
params.kmeans_data = "data/kmeans_input.txt"
params.matmul_a    = "data/matrix_a.txt"
params.matmul_b    = "data/matrix_b.txt"
params.lysozyme_pdb = "data/lysozyme.pdb"


process KMEANS {
    tag "KMeans"
    publishDir "results/kmeans", mode: 'copy'

    input:
    val data

    output:
    path "kmeans_results.txt"

    script:
    """
    echo "Executando KMeans em $data..."
    # Substitua pelo seu comando real, ex: python kmeans.py $data
    echo "Cluster 1: [x,y], Cluster 2: [z,w]" > kmeans_results.txt
    """
}


workflow {
    KMEANS(params.kmeans_data)
}
