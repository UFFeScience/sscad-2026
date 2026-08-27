#!/usr/bin/env python3
import parsl
from parsl.app.app import python_app
from parsl.configs.local_threads import config
import os

# Configuração do Parsl
parsl.load(config)

@python_app
def generate_data(n_samples=1000, n_features=2):
    import numpy as np
    from sklearn.datasets import make_blobs
    X, _ = make_blobs(n_samples=n_samples, n_features=n_features, centers=3, random_state=42)
    return X

@python_app
def compute_kmeans(data, n_clusters=3):
    from sklearn.cluster import KMeans
    import numpy as np
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    kmeans.fit(data)
    return kmeans.cluster_centers_

def main():
    print("Iniciando Workflow KMeans...")
    
    # Passo 1: Gerar dados
    data_future = generate_data(n_samples=2000)
    
    # Passo 2: Calcular Centroides (depende do data_future)
    centers_future = compute_kmeans(data_future, n_clusters=3)
    
    # Resultado
    centers = centers_future.result()
    print("Centroides encontrados:")
    print(centers)
    
    parsl.clear()

if __name__ == '__main__':
    main()