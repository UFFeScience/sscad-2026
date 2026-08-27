"""
PyCOMPSs workflows para KMeans, Matmul e Lysozyme (GROMACS).

Execução:
    runcompss ./pycompss_workflows.py --algorithm kmeans   # ou matmul ou lysozyme

O GROMACS (gmx e gmx_mpi) deve estar disponível no PATH do ambiente de execução.
"""



import os



import time



import argparse



import subprocess



from pathlib import Path



from pycompss.api.task import task



from pycompss.api.api import compss_wait_on



import numpy as np



@task(returns=np.ndarray)
def kmeans_task(data, k, max_iter=100, tol=1e-4):
    """
    Executa o algoritmo KMeans nos dados (lista de vetores) com k clusters.
    Retorna os centróides como um array NumPy.
    """
    X = np.array(data)
    n_samples = X.shape[0]
    # Inicialização aleatória – escolhe amostras distintas
    idx = np.random.choice(n_samples, k, replace=False)
    centroids = X[idx]

    for _ in range(max_iter):
        # Distâncias euclidiana para o centróide mais próximo
        distances = np.linalg.norm(X[:, None] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        new_centroids = np.array([X[labels == j].mean(axis=0) for j in range(k)])
        if np.linalg.norm(new_centroids - centroids) < tol:
            centroids = new_centroids
            break
        centroids = new_centroids
    return centroids



def ensure_kmeans_dataset(path):
    """Cria um pequeno conjunto de dados CSV se o arquivo não existir."""
    if Path(path).is_file():
        return
    np.random.seed(0)
    data = np.random.randn(100, 2) * 10 + 5   # 100 pontos 2‑D
    np.savetxt(path, data, delimiter=',')
    print(f"Dataset KMeans criado em {path}")



def workflow_kmeans(args):
    start = time.time()
    ensure_kmeans_dataset(args.kmeans_data)

    # Carrega os dados
    data = np.loadtxt(args.kmeans_data, delimiter=',')

    # Executa a tarefa KMeans (single‑task)
    centroids = kmeans_task(data.tolist(), args.k)
    centroids = compss_wait_on(centroids)

    print("Centróides KMeans:")
    print(centroids)
    print(f"Tempo total: {time.time() - start:.2f} s")



def parse_args():
    parser = argparse.ArgumentParser(
        description="PyCOMPSs workflows para KMeans, Matmul e Lysozyme"
    )
    parser.add_argument(
        "--algorithm",
        choices=["kmeans", "matmul", "lysozyme"],
        required=True,
        help="Qual algoritmo executar"
    )
    parser.add_argument(
        "--out-dir",
        default="./results",
        help="Diretório para arquivos de saída (padrão: ./results)"
    )

    # KMeans
    parser.add_argument(
        "--kmeans-data",
        default="./kmeans_data.csv",
        help="Arquivo CSV com o dataset"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Número de clusters"
    )

    # Matmul
    parser.add_argument(
        "--matmul-A",
        default="./matrix_A.npy",
        help="Arquivo .npy da matriz A"
    )
    parser.add_argument(
        "--matmul-B",
        default="./matrix_B.npy",
        help="Arquivo .npy da matriz B"
    )

    # Lysozyme
    parser.add_argument(
        "--pdb",
        default="./lysozyme.pdb",
        help="Arquivo PDB de entrada"
    )
    parser.add_argument(
        "--mpi-ranks",
        type=int,
        default=1,
        help="Número de processos MPI para mdrun"
    )
    return parser.parse_args()



def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.algorithm == "kmeans":
        workflow_kmeans(args)
    elif args.algorithm == "matmul":
        workflow_matmul(args)
    elif args.algorithm == "lysozyme":
        workflow_lysozyme(args)



if __name__ == "__main__":
    main()
