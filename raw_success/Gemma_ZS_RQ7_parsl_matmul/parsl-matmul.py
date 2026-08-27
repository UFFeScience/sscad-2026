import time
import numpy as np
import argparse
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
    # Em Parsl, retornamos o resultado pois não há memória compartilhada mutável entre nós
    return C + np.dot(A, B)

def dot(A, B, C):
    n, m = len(A), len(B[0])
    for i in range(n):
        for j in range(m):
            for k in range(n):
                # Atualiza o bloco C[i][j] com o resultado da tarefa
                C[i][j] = fused_multiply_add(A[i][k], B[k][j], C[i][j]).result()

def main(num_blocks, elems_per_block, seed):
    start_time = time.time()
    A, B, C = [], [], []
    
    for i in range(num_blocks):
        A.append([]); B.append([]); C.append([])
        bid = 0
        for j in range(num_blocks):
            A[-1].append(generate_block(elems_per_block, num_blocks, seed=seed + bid))
            bid += 1
            B[-1].append(generate_block(elems_per_block, num_blocks, seed=seed + bid))
            bid += 1
            C[-1].append(generate_block(elems_per_block, num_blocks, set_to_zero=True))

    # Resolve a geração inicial
    A = [[f.result() for f in row] for row in A]
    B = [[f.result() for f in row] for row in B]
    C = [[f.result() for f in row] for row in C]
    
    initialization_time = time.time()
    dot(A, B, C)
    multiplication_time = time.time()

    print(f"Initialization time: {initialization_time - start_time:.4f}")
    print(f"Multiplication time: {multiplication_time - initialization_time:.4f}")
    print(f"Total time: {multiplication_time - start_time:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--num_blocks', type=int, default=1)
    parser.add_argument('-e', '--elems_per_block', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    opts = parser.parse_args()
    main(**vars(opts))