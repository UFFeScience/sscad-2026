import os



import time



import argparse



import numpy as np



from pycompss.api.task import task



from pycompss.api.api import compss_wait_on



@task(returns=list)
def multiply_matrix_chunk(matrix_a, matrix_b, row_idx):
    """Calcula uma linha da matriz resultante."""
    row_a = matrix_a[row_idx]
    result_row = []
    for j in range(len(matrix_b[0])):
        col_b = [matrix_b[i][j] for i in range(len(matrix_b))]
        result_row.append(sum(row_a[k] * col_b[k] for k in range(len(col_b))))
    return result_row



def run_matmul(matrix_a, matrix_b):
    print("Iniciando MatMul...")
    rows_a = len(matrix_a)
    results = []
    for i in range(rows_a):
        results.append(multiply_matrix_chunk(matrix_a, matrix_b, i))
    
    # O resultado final é a lista de linhas
    return compss_wait_on(results)



def main():
    # --- Teste MatMul (unico algoritmo definido neste arquivo) ---
    mat_a = [[1, 2], [3, 4]]
    mat_b = [[5, 6], [7, 8]]
    res_matmul = run_matmul(mat_a, mat_b)
    print(f"MatMul Result: {res_matmul}")



if __name__ == "__main__":
    main()
