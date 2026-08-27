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
def kmeans_app(data, n_clusters, random_state=42):
    """Aplica KMeans nos dados fornecidos."""
    from sklearn.cluster import KMeans
    import numpy as np
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(data)
    return labels



def run_kmeans_workflow():
    """Executa o workflow KMeans."""
    # Gera dados sintéticos
    X = np.random.rand(100, 2)
    labels_future = kmeans_app(X, n_clusters=3)
    labels = labels_future.result()
    print("KMeans labels:", labels[:5], "...")
    return labels



if __name__ == "__main__":
    # Executa apenas o workflow KMeans (matmul/lysozyme nao estao definidos
    # neste arquivo -- cada algoritmo foi gerado em um arquivo separado)
    print("=== Workflow KMeans ===")
    run_kmeans_workflow()
