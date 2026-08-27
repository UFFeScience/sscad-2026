import os
import subprocess
import numpy as np
import argparse
from pycompss.api.task import task
from pycompss.api.api import compss_wait_on

# ==============================================================================
# WORKFLOW: MATRIX MULTIPLICATION (Matmul)
# Estratégia: Dividir a matriz A em blocos de linhas e calcular a multiplicação
# em paralelo.
# ==============================================================================

@task(returns=np.ndarray)
def multiply_block(matrix_a_block, matrix_b):
    """Calcula a multiplicação de um bloco da matriz A pela matriz B."""
    return np.dot(matrix_a_block, matrix_b)

def run_matmul(A, B):
    print("Iniciando Workflow Matmul...")
    # Divide a matriz A em N blocos (onde N é o número de workers disponíveis)
    num_blocks = 4
    chunks = np.array_split(A, num_blocks)

    partial_results = []
    for chunk in chunks:
        partial_results.append(multiply_block(chunk, B))

    # Concatena os resultados (Sincronização)
    result = np.vstack(compss_wait_on(partial_results))
    return result

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='PyCOMPSs Matmul Workflow')
    parser.parse_args()

    # Matrizes dummy
    A = np.random.rand(100, 50)
    B = np.random.rand(50, 80)
    res = run_matmul(A, B)
    print("Matmul concluído. Shape do resultado:", res.shape)

if __name__ == "__main__":
    main()
