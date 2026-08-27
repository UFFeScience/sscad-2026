#!/usr/bin/env python3
import parsl
from parsl.app.app import python_app
from parsl.configs.local_threads import config

parsl.load(config)

@python_app
def create_matrix(size):
    import numpy as np
    return np.random.rand(size, size)

@python_app
def multiply_matrices(A, B):
    import numpy as np
    return np.dot(A, B)

def main():
    print("Iniciando Workflow Matmul...")
    size = 500
    
    # Criar duas matrizes em paralelo
    future_a = create_matrix(size)
    future_b = create_matrix(size)
    
    # Multiplicar (aguarda os futures a e b)
    result_future = multiply_matrices(future_a, future_b)
    
    # Obter resultado e calcular a norma para validação
    import numpy as np
    res = result_future.result()
    print(f"Matriz resultante {size}x{size} calculada.")
    print(f"Norma de Frobenius do resultado: {np.linalg.norm(res)}")
    
    parsl.clear()

if __name__ == '__main__':
    main()