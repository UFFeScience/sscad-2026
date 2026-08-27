#!/usr/bin/env python3
"""
Workflow Parsl para multiplicação de matrizes usando numpy.
Gera duas matrizes aleatórias, multiplica em blocos e combina os resultados.
"""

import os
import numpy as np
import parsl
from parsl.app.app import python_app
from parsl.configs.local_threads import config

os.makedirs('output', exist_ok=True)

@python_app
def generate_matrix(shape, seed):
    """Gera uma matriz aleatória."""
    np.random.seed(seed)
    return np.random.rand(*shape)

@python_app
def multiply_block(A_block, B_block):
    """Multiplica dois blocos de matrizes."""
    return np.dot(A_block, B_block)

@python_app
def assemble_matrix(blocks, n_blocks, block_size):
    """Monta a matriz resultante a partir dos blocos."""
    rows, cols = n_blocks
    result = np.zeros((rows * block_size, cols * block_size))
    for i in range(rows):
        for j in range(cols):
            result[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = blocks[i][j]
    return result

def main():
    parsl.load(config)

    # Configurações
    n = 1000                # tamanho da matriz
    block_size = 100        # tamanho do bloco
    n_blocks = n // block_size  # número de blocos em cada dimensão

    # Gera matrizes A e B
    A = generate_matrix((n, n), seed=0).result()
    B = generate_matrix((n, n), seed=1).result()
    print("Matrizes A e B geradas.")

    # Divide em blocos
    A_blocks = [[A[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
                 for j in range(n_blocks)] for i in range(n_blocks)]
    B_blocks = [[B[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
                 for j in range(n_blocks)] for i in range(n_blocks)]

    # Multiplica cada bloco de A com cada bloco de B (produto externo)
    block_results = {}
    for i in range(n_blocks):
        for k in range(n_blocks):
            for j in range(n_blocks):
                # Multiplica A[i][k] * B[k][j] e acumula em C[i][j]
                if (i, j) not in block_results:
                    block_results[(i, j)] = []
                block_results[(i, j)].append(multiply_block(A_blocks[i][k], B_blocks[k][j]))

    # Combina os blocos acumulados
    final_blocks = {}
    for (i, j), apps in block_results.items():
        partials = [app.result() for app in apps]
        final_blocks[(i, j)] = sum(partials)

    # Monta a matriz final -- final_blocks e um dict chaveado por (i, j), mas
    # assemble_matrix espera uma lista aninhada indexada blocks[i][j]
    final_blocks_list = [[final_blocks[(i, j)] for j in range(n_blocks)] for i in range(n_blocks)]
    C = assemble_matrix(final_blocks_list, (n_blocks, n_blocks), block_size).result()

    # Verifica com numpy (opcional)
    C_np = np.dot(A, B)
    print("Diferença máxima:", np.max(np.abs(C - C_np)))

    # Salva resultado
    np.savetxt('output/C_matrix.txt', C)
    print("Matriz C salva em output/C_matrix.txt")

    parsl.clear()

if __name__ == '__main__':
    main()