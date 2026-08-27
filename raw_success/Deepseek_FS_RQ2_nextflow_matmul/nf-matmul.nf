#!/usr/bin/env nextflow

/*
 * pipeline input parameters
 */
params.matrix_size = 100
params.outdir = "results"

/*
 * MATMUL process - Executa multiplicação de matrizes
 */
process MATMUL {
    publishDir params.outdir, mode: 'copy'
    cpus 2
    
    input:
    val matrix_size
    
    output:
    path "matmul_results.txt"
    path "A.csv"
    path "B.csv"
    path "matrix_result.csv"
    
    script:
    """
    python3 <<EOF
import numpy as np
import time
import sys

# Gerar matrizes aleatórias
n = ${matrix_size}
A = np.random.rand(n, n)
B = np.random.rand(n, n)

# Medir tempo de multiplicação
start_time = time.time()
C = np.matmul(A, B)
end_time = time.time()
elapsed_time = end_time - start_time

# Salvar matrizes
np.savetxt('matrix_a.csv', A, delimiter=',')
np.savetxt('matrix_b.csv', B, delimiter=',')
np.savetxt('matrix_result.csv', C, delimiter=',')

# Salvar resultados
with open('matmul_results.txt', 'w') as f:
    f.write(f'Matrix Multiplication Results\\n')
    f.write(f'============================\\n')
    f.write(f'Matrix size: {n}x{n}\\n')
    f.write(f'Execution time: {elapsed_time:.6f} seconds\\n')
    f.write(f'Result matrix shape: {C.shape}\\n')
    f.write(f'Result matrix stats: min={C.min():.4f}, max={C.max():.4f}, mean={C.mean():.4f}\\n')
EOF
    """
}

workflow {
    log.info """\
        M A T M U L   P I P E L I N E
        =============================
        matrix size  : ${params.matrix_size}
        outdir       : ${params.outdir}
        """.stripIndent(true)
    
    MATMUL(params.matrix_size)
}