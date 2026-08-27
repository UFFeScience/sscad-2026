import parsl



from parsl.app.app import python_app



from parsl.configs.local_threads import config as local_config



parsl.load(local_config)



@python_app
def run_kmeans(n_samples=10000, n_clusters=5):
    """Simula o algoritmo KMeans."""
    from sklearn.cluster import KMeans
    import numpy as np
    print(f"Iniciando KMeans com {n_samples} amostras...")
    X = np.random.rand(n_samples, 2)
    kmeans = KMeans(n_clusters=n_clusters, n_init='auto').fit(X)
    return f"KMeans concluído. Centros calculados: {len(kmeans.cluster_centers_)}"



def main():
    print("Iniciando Workflow Paralelo via Parsl...\n")

    future_kmeans = run_kmeans(50000)

    # Coletando resultados
    print("Aguardando finalização das tarefas...")
    print(f"Resultado KMeans: {future_kmeans.result()}")



if __name__ == "__main__":
    main()
