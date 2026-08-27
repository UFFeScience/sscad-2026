import compss



import subprocess



def matmul_task(matrix_a, matrix_b):
    """
    Simula a multiplicação de matrizes. 
    Em um cenário real, você usaria numpy ou bibliotecas BLAS.
    """
    import numpy as np
    res = np.dot(matrix_a, matrix_b)
    return res



def workflow_matmul():
    print("Iniciando Workflow MatMul...")
    # Definindo dados de entrada (Exemplo: matrizes 100x100)
    import numpy as np
    a = np.random.rand(100, 100)
    b = np.random.rand(100, 100)
    
    # A tarefa é enviada ao COMPSs
    result = matmul_task(a, b)
    return result



if __name__ == "__main__":
    # O COMPSs requer que as funções sejam decoradas ou chamadas dentro de um contexto
    # dependendo da versão. No PyCOMPSs moderno, basta chamar as funções.

    # MatMul (unico algoritmo definido neste arquivo)
    res_matmul = workflow_matmul()
    print("MatMul concluído.")
