"""
Exemplo de uso do Parsl para criar workflows de:
- KMeans (clustering)
- Matmul (multiplicação de matrizes)
- Lysozyme (simulação molecular com GROMACS)

Requer: parsl, numpy, scikit-learn, e GROMACS instalados (gmx e gmx_mpi).
"""



import os



import numpy as np



import parsl



from parsl import python_app, bash_app



from parsl.config import Config



from parsl.executors import ThreadPoolExecutor



config = Config(
    executors=[ThreadPoolExecutor(max_threads=4)]
)



parsl.load(config)



@python_app
def matmul_app(A, B):
    """Multiplica duas matrizes usando numpy."""
    import numpy as np
    return np.dot(A, B)



def run_matmul_workflow():
    """Executa o workflow de multiplicação de matrizes."""
    A = np.random.rand(10, 10)
    B = np.random.rand(10, 10)
    result_future = matmul_app(A, B)
    result = result_future.result()
    print("Matmul result shape:", result.shape)
    return result



if __name__ == "__main__":
    # Executa apenas o workflow Matmul (kmeans/lysozyme nao estao definidos
    # neste arquivo -- cada algoritmo foi gerado em um arquivo separado)
    print("\n=== Workflow Matmul ===")
    run_matmul_workflow()
