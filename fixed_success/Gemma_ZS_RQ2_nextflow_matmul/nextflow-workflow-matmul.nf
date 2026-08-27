nextflow.enable.dsl=2

// Parâmetros de entrada
params.kmeans_data = "data/kmeans_input.txt"
params.matmul_a    = "data/matrix_a.txt"
params.matmul_b    = "data/matrix_b.txt"
params.lysozyme_pdb = "data/lysozyme.pdb"


process MATMUL {
    tag "MatMul"
    publishDir "results/matmul", mode: 'copy'

    input:
    val mat_a
    val mat_b

    output:
    path "matmul_result.txt"

    script:
    """
    echo "Multiplicando matrizes $mat_a e $mat_b..."
    # Substitua pelo seu comando real, ex: ./matmul $mat_a $mat_b
    echo "Resultado da Multiplicação de Matrizes" > matmul_result.txt
    """
}


workflow {
    MATMUL(params.matmul_a, params.matmul_b)
}
