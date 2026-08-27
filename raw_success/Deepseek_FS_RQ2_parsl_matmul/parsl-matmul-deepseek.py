#!/usr/bin/env python3
import os
import parsl
from parsl.app.app import python_app
from parsl.configs.local_threads import config
import numpy as np

def main():
    # Criar diretório de saída
    os.makedirs('output', exist_ok=True)
    
    # Carregar configuração do Parsl
    parsl.load(config)
    
    @python_app
    def generate_matrix_app(rows, cols, random_state=42):
        """Gerar uma matriz aleatória"""
        np.random.seed(random_state)
        return np.random.randn(rows, cols).tolist()
    
    @python_app
    def matmul_app(A, B):
        """Multiplicação de matrizes"""
        import numpy as np
        A = np.array(A)
        B = np.array(B)
        return np.dot(A, B).tolist()
    
    @python_app
    def matmul_transpose_app(A, B):
        """Multiplicação de A * B^T"""
        import numpy as np
        A = np.array(A)
        B = np.array(B)
        return np.dot(A, B.T).tolist()
    
    # Tamanhos das matrizes
    size = 100
    print(f"Gerando matrizes de tamanho {size}x{size}...")
    
    # Gerar matrizes em paralelo
    A_future = generate_matrix_app(size, size, random_state=42)
    B_future = generate_matrix_app(size, size, random_state=43)
    
    A = A_future.result()
    B = B_future.result()
    
    print("Calculando multiplicação de matrizes...")
    
    # Multiplicação normal
    C_future = matmul_app(A, B)
    C = np.array(C_future.result())
    
    print(f"Resultado da multiplicação: matriz {C.shape[0]}x{C.shape[1]}")
    
    # Multiplicação com transposição
    D_future = matmul_transpose_app(A, B)
    D = np.array(D_future.result())
    
    print(f"Resultado de A * B^T: matriz {D.shape[0]}x{D.shape[1]}")
    
    # Salvar resultados
    import json
    results = {
        'size': size,
        'A_shape': [size, size],
        'B_shape': [size, size],
        'C_shape': list(C.shape),
        'D_shape': list(D.shape),
        'C_sample': C[:3, :3].tolist(),  # Amostra pequena para não gerar arquivo enorme
        'D_sample': D[:3, :3].tolist()
    }
    
    with open('output/matmul_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Resultados salvos em output/matmul_results.json")
    
    # Limpar Parsl
    parsl.clear()

if __name__ == '__main__':
    main()