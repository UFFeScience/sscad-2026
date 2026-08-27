import os
import subprocess
import numpy as np
import argparse
from pycompss.api.task import task
from pycompss.api.api import compss_wait_on

# ==============================================================================
# WORKFLOW: K-MEANS
# Estratégia: Paralelizar o cálculo de distância e atribuição de clusters.
# ==============================================================================

@task(returns=np.ndarray)
def assign_clusters(data_chunk, centroids):
    """Atribui cada ponto do chunk ao centroide mais próximo."""
    # Calcula a distância Euclidiana: sqrt(sum((a-b)^2))
    distances = np.linalg.norm(data_chunk[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)

@task(returns=np.ndarray)
def update_centroids(data, assignments, k):
    """Calcula a nova posição dos centroides baseada nas atribuições."""
    new_centroids = np.array([data[assignments == i].mean(axis=0) for i in range(k)])
    return new_centroids

def run_kmeans(data, k=3, iterations=5):
    print("Iniciando Workflow KMeans...")
    # Inicialização simples de centroides
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]

    for i in range(iterations):
        num_chunks = 4
        chunks = np.array_split(data, num_chunks)

        # Passo 1: Atribuição paralela
        partial_assignments = []
        for chunk in chunks:
            partial_assignments.append(assign_clusters(chunk, centroids))

        # Sincroniza atribuições
        assignments = np.concatenate(compss_wait_on(partial_assignments))

        # Passo 2: Atualização dos centroides
        centroids = compss_wait_on(update_centroids(data, assignments, k))

    return centroids

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='PyCOMPSs KMeans Workflow')
    parser.parse_args()

    # Dados dummy
    data = np.random.rand(1000, 2)
    res = run_kmeans(data, k=3)
    print("KMeans concluído. Centroides finais:\n", res)

if __name__ == "__main__":
    main()
