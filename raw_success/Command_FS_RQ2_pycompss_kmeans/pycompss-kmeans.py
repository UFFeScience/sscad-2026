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

def generate_kmeans_data(n_samples=1000, n_features=2, n_clusters=3):
    """Gera dados aleatórios para o algoritmo KMeans."""
    data = []
    for _ in range(n_samples):
        point = [random.random() * 10 for _ in range(n_features)]
        data.append(point)
    return data

# -----------------------------------------------------------------------------
# TAREFAS (WORKFLOW) DEFINIDAS COM PYCOMPS
# -----------------------------------------------------------------------------

@task(returns=list)
def kmeans_task(data, k, max_iters=10):
    """
    Executa o algoritmo KMeans.
    :param data: Lista de pontos (listas).
    :param k: Número de clusters.
    :param max_iters: Iterações máximas.
    :return: Lista de centróides.
    """
    # Inicialização simples: seleciona k pontos aleatórios como centróides
    centroids = random.sample(data, min(k, len(data)))

    for _ in range(max_iters):
        clusters = {i: [] for i in range(k)}

        # Atribuição de pontos aos centróides mais próximos
        for point in data:
            distances = [sum((p - c) ** 2 for p, c in zip(point, centroid)) ** 0.5
                         for centroid in centroids]
            closest = distances.index(min(distances))
            clusters[closest].append(point)

        # Atualização dos centróides
        new_centroids = []
        for i in range(k):
            if clusters[i]:
                # Calcula a média dos pontos no cluster
                n_features = len(data[0])
                mean = [sum(p[j] for p in clusters[i]) / len(clusters[i]) for j in range(n_features)]
                new_centroids.append(mean)
            else:
                # Mantém o antigo se o cluster estiver vazio
                new_centroids.append(centroids[i])

        centroids = new_centroids

    return centroids

# -----------------------------------------------------------------------------
# FUNÇÃO PRINCIPAL (ORQUESTRAÇÃO)
# -----------------------------------------------------------------------------

def main():
    start_time = time.time()

    # 1. Preparar ambiente
    workspace = ensure_working_directory()
    print(f"Diretório de trabalho: {workspace}\n")

    # 2. Gerar dados para KMeans
    print("--- Iniciando Workflow KMeans ---")
    kmeans_data = generate_kmeans_data(n_samples=500, n_clusters=4)
    # Enviar dados para tarefa (PyCOMPSs gerencia a transferência)
    kmeans_centroids = kmeans_task(kmeans_data, k=4)
    # Esperar resultado
    kmeans_centroids = compss_wait_on(kmeans_centroids)
    print(f"Centróides KMeans calculados: {len(kmeans_centroids)}")
    print(f"Primeiro centróide: {kmeans_centroids[0]}\n")

    # 3. Finalização
    total_time = time.time() - start_time

    print("=== Workflow concluído ===")
    print(f"Tempo total: %f segundos" % total_time)

def parse_args():
    """
    Arguments parser.
    """
    description = 'PyCOMPSs Workflow Demo: KMeans'
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
