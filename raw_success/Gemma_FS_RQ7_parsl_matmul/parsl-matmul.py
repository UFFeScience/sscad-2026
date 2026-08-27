import numpy as np
import time
import parsl
from parsl.app.app import python_app
from parsl.configs.local_threads import config

parsl.load(config)

@python_app
def generate_block(size, num_blocks, seed=0, set_to_zero=False):
    np.random.seed(seed)
    if not set_to_zero:
        b = np.random.random((size, size))
        b /= np.sum(b) * float(num_blocks)
    else:
        b = np.zeros((size, size))
    return b

@python_app
def fused_multiply_add(A, B, C):
    # No Parsl, retornamos o novo valor de C para criar a dependência
    return C + np.dot(A, B)

def dot(A, B, C):
    n, m = len(A), len(B[0])
    for i in range(n):
        for j in range(m):
            for k in range(n):
                # Sobrescrevemos o Future de C[i][j] com o novo Future retornado
                C[i][j] = fused_multiply_add(A[i][k], B[k][j], C[i][j])

def main(num_blocks, elems_per_block, seed):
    A, B, C = [], [], []
    for i in range(num_blocks):
        A.append([]); B.append([]); C.append([])
        for j in range(num_blocks):
            A[i].append(generate_block(elems_per_block, num_blocks, seed=seed + (i*j)))
            B[i].append(generate_block(elems_per_block, num_blocks, seed=seed + (i*j) + 1))
            C[i].append(generate_block(elems_per_block, num_blocks, set_to_zero=True))

    dot(A, B, C)
    
    # Sincronização: esperar o último bloco da matriz C
    [ [col.result() for col in row] for row in C ]
    print("Matmul completa.")

if __name__ == "__main__":
    main(2, 4, 0)