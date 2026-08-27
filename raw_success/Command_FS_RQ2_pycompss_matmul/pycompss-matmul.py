import os
import time
import argparse
import random
import subprocess
import sys
from pprint import pprint

from pycompss.api.task import task
from pycompss.api.api import compss_wait_on
from pycompss.api.parameter import Type, DIRECTION

# -----------------------------------------------------------------------------
# FUNÇÕES AUXILIARES PARA CONFIGURAÇÃO E DADOS
# -----------------------------------------------------------------------------

def ensure_working_directory():
    """Cria um diretório de trabalho temporário se não existir."""
    working_dir = os.path.join(os.getcwd(), "compss_workspace")
    os.makedirs(working_dir, exist_ok=True)
    return working_dir

def generate_matrices(dim=100):
    """Gera duas matrizes aleatórias para multiplicação."""
    # Usando inteiros pequenos para evitar estouro e facilitar a verificação
    A = [[random.randint(1, 9) for _ in range(dim)] for _ in range(dim)]
    B = [[random.randint(1, 9) for _ in range(dim)] for _ in range(dim)]
    return A, B

# -----------------------------------------------------------------------------
# TAREFAS (WORKFLOW) DEFINIDAS COM PYCOMPS
# -----------------------------------------------------------------------------

@task(returns=list)
def matmul_task(A, B):
    """
    Executa a multiplicação de matrizes (Matmul).
    :param A: Matriz A (lista de listas).
    :param B: Matriz B (lista de listas).
    :return: Matriz resultante (lista de listas).
    """
    n = len(A)
    m = len(B[0])
    p = len(B)

    # Inicializa a matriz de resultados com zeros
    result = [[0 for _ in range(m)] for _ in range(n)]

    for i in range(n):
        for j in range(m):
            for k in range(p):
                result[i][j] += A[i][k] * B[k][j]

    return result

# -----------------------------------------------------------------------------
# FUNÇÃO PRINCIPAL (ORQUESTRAÇÃO)
# -----------------------------------------------------------------------------

def main():
    start_time = time.time()

    # 1. Preparar ambiente
    workspace = ensure_working_directory()
    print(f"Diretório de trabalho: {workspace}\n")

    # 2. Gerar dados para Matmul
    print("--- Iniciando Workflow Matmul ---")
    mat_dim = 50 # Pequeno o suficiente para ser rápido
    A, B = generate_matrices(dim=mat_dim)
    mat_result = matmul_task(A, B)
    mat_result = compss_wait_on(mat_result)
    # Verificação simples da soma de um elemento
    total_sum = sum(sum(row) for row in mat_result)
    print(f"Resultado Matmul calculado. Soma dos elementos: {total_sum}\n")

    # 3. Finalização
    total_time = time.time() - start_time

    print("=== Workflow concluído ===")
    print(f"Tempo total: %f segundos" % total_time)

def parse_args():
    """
    Arguments parser.
    """
    description = 'PyCOMPSs Workflow Demo: Matmul'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        '-d', '--dataset', type=str,
        default=os.path.join(os.getcwd(), 'dataset'),
        help='Diretório para dados (padrão: ./dataset)'
    )
    return parser.parse_args()

if __name__ == "__main__":
    # Nota: Para executar com o PyCOMPSs, normalmente se usa 'runcompss script.py'
    # Este bloco permite a execução local para teste.
    opts = parse_args()
    main()
