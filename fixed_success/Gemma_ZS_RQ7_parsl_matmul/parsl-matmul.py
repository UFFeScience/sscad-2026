import time
import numpy as np
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
    # No Parsl, retornamos o resultado pois não há mutação compartilhada de memória
    return C + np.dot(A, B)

def dot(A, B, C):
    n, m = len(A), len(B[0])
    for i in range(n):
        for j in range(m):
            # O loop 'k' é sequencial aqui para evitar race conditions no bloco C[i][j]
            # mas cada FMA pode ser disparado como tarefa
            for k in range(n):
                # Aguardamos o resultado de C para a próxima soma (dependência de dados)
                C[i][j] = fused_multiply_add(A[i][k], B[k][j], C[i][j]).result()

def main(num_blocks, elems_per_block, seed):
    start_time = time.time()
    A, B, C = [], [], []
    
    for i in range(num_blocks):
        row_A, row_B, row_C = [], [], []
        bid = 0
        for j in range(num_blocks):
            row_A.append(generate_block(elems_per_block, num_blocks, seed=seed + bid))
            bid += 1
            row_B.append(generate_block(elems_per_block, num_blocks, seed=seed + bid))
            bid += 1
            row_C.append(generate_block(elems_per_block, num_blocks, set_to_zero=True))
        A.append(row_A); B.append(row_B); C.append(row_C)

    # Resolver todos os blocos iniciais
    A = [[f.result() for f in row] for row in A]
    B = [[f.result() for f in row] for row in B]
    C = [[f.result() for f in row] for row in C]
    
    initialization_time = time.time()
    dot(A, B, C)
    multiplication_time = time.time()

    print(f"Initialization time: {initialization_time - start_time:.4f}")
    print(f"Multiplication time: {multiplication_time - initialization_time:.4f}")

if __name__ == "__main__":
    main(2, 2, 0)