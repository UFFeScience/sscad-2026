import os



import time



import argparse



import numpy as np



from pycompss.api.task import task



from pycompss.api.api import compss_wait_on



@task(returns=list)
def compute_centroids(data, centroids):
    """Calcula a média dos pontos atribuídos a cada centroide."""
    new_centroids = []
    for c in centroids:
        points = [p for p in data if min([np.linalg.norm(np.array(p) - np.array(c)) for c in centroids]) == np.linalg.norm(np.array(p) - np.array(c))]
        if points:
            new_centroids.append(np.mean(points, axis=0).tolist())
        else:
            new_centroids.append(c)
    return new_centroids



def run_kmeans(data, k=3, iterations=5):
    print("Iniciando KMeans...")
    # Inicialização simples: pega os primeiros k pontos
    centroids = data[:k]
    for i in range(iterations):
        centroids = compute_centroids(data, centroids)
    return compss_wait_on(centroids)



def main():
    # --- Teste KMeans (unico algoritmo definido neste arquivo) ---
    data_kmeans = [[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]]
    res_kmeans = run_kmeans(data_kmeans)
    print(f"KMeans Centroids: {res_kmeans}")



if __name__ == "__main__":
    main()
