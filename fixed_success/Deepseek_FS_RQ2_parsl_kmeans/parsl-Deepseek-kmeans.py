#!/usr/bin/env python3
"""
Workflow Parsl para calcular KMeans em dados sintéticos.
Gera um arquivo de saída com os centróides e rótulos.
"""

import os
import parsl
from parsl.app.app import python_app
from parsl.configs.local_threads import config

# Carrega configuração Parsl (execução local com threads)
parsl.load(config)

@python_app
def kmeans_fit(data, n_clusters=3, random_state=42):
    """Aplica KMeans e retorna os centróides e rótulos."""
    from sklearn.cluster import KMeans
    import numpy as np

    km = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = km.fit_predict(data)
    centroids = km.cluster_centers_
    return centroids, labels

def main():
    # Gera dados sintéticos (100 pontos, 2 features)
    import numpy as np
    np.random.seed(0)
    data = np.random.rand(100, 2)

    # Executa KMeans como app Parsl
    result_future = kmeans_fit(data, n_clusters=3)

    # Aguarda resultados (kmeans_fit e um unico python_app -- retorna uma
    # unica AppFuture cujo .result() e a tupla (centroids, labels), nao duas
    # futures separadas para desempacotar)
    centroids, labels = result_future.result()

    # Salva resultados
    os.makedirs('output', exist_ok=True)
    with open('output/kmeans_centroids.txt', 'w') as f:
        f.write("Centróides:\n")
        for c in centroids:
            f.write(f"{c[0]:.4f} {c[1]:.4f}\n")
    with open('output/kmeans_labels.txt', 'w') as f:
        f.write("Rótulos:\n")
        for l in labels:
            f.write(f"{l}\n")

    print("KMeans concluído. Centróides:")
    print(centroids)

if __name__ == '__main__':
    main()
    parsl.clear()