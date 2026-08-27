import os



import time



import argparse



import subprocess



from pprint import pprint



from pycompss.api.task import task



from pycompss.api.api import compss_wait_on



@task(returns=list)
def kmeans_partial(points_chunk, centroids):
    """
    Executa uma iteração parcial do KMeans em um subconjunto de pontos.
    Retorna [novos_centroides, contagem_por_cluster] como uma lista.
    """
    # Inicializa acumuladores
    k = len(centroids)
    sums = [[0.0] * len(points_chunk[0]) for _ in range(k)]
    counts = [0] * k

    # Atribui cada ponto ao centróide mais próximo e acumula
    for point in points_chunk:
        # Calcula distância (exemplo: euclidiana)
        min_dist = float('inf')
        cluster = -1
        for i, c in enumerate(centroids):
            dist = sum((p - q) ** 2 for p, q in zip(point, c)) ** 0.5
            if dist < min_dist:
                min_dist = dist
                cluster = i
        sums[cluster] = [s + p for s, p in zip(sums[cluster], point)]
        counts[cluster] += 1

    # Calcula novos centroides parciais (média dos pontos no cluster)
    new_centroids = []
    for i in range(k):
        if counts[i] > 0:
            new_centroids.append([s / counts[i] for s in sums[i]])
        else:
            new_centroids.append(centroids[i])  # mantém o anterior se vazio
    return [new_centroids, counts]



@task(returns=list)
def kmeans_merge(partial_results):
    """
    Combina resultados parciais de kmeans_partial.
    Entrada: lista de [centroides_parciais, contagens_parciais]
    Saída: [centroides_finais, contagens_totais]
    """
    k = len(partial_results[0][0])
    dim = len(partial_results[0][0][0])
    total_counts = [0] * k
    total_sums = [[0.0] * dim for _ in range(k)]

    for centroids_partial, counts_partial in partial_results:
        for i in range(k):
            # Cada centroide parcial já é uma média, mas precisamos do somatório
            # para combinar corretamente. Vamos reconstruir o somatório.
            # NOTE (rerun-simple fix): counts_partial[i] é um escalar (a
            # contagem do cluster i), não uma lista -- o zip() original
            # tentava iterar sobre ele por dimensão junto com
            # total_sums[i]/centroids_partial[i] (TypeError: 'int' object
            # is not iterable). Corrigido para multiplicar cada componente
            # do centroide pelo escalar counts_partial[i], não fazer zip
            # com ele.
            c = counts_partial[i]
            total_sums[i] = [s + c * p for s, p in zip(total_sums[i],
                                                         centroids_partial[i])]
            total_counts[i] += counts_partial[i]

    final_centroids = []
    for i in range(k):
        if total_counts[i] > 0:
            final_centroids.append([s / total_counts[i] for s in total_sums[i]])
        else:
            final_centroids.append([0.0] * dim)
    return [final_centroids, total_counts]



def main_kmeans(num_points, num_clusters, dim, num_chunks):
    """Gera e executa o workflow KMeans."""
    start = time.time()

    # Gera dados de exemplo (aleatórios)
    import random
    random.seed(42)
    all_points = [[random.random() for _ in range(dim)] for _ in range(num_points)]

    # Divide em chunks
    chunk_size = num_points // num_chunks
    chunks = [all_points[i:i + chunk_size] for i in range(0, num_points, chunk_size)]
    if len(chunks) > num_chunks:
        chunks = chunks[:num_chunks]  # garante número exato

    # Centroides iniciais (primeiros k pontos)
    centroids = all_points[:num_clusters]

    # Executa iterações (apenas 1 para demonstração)
    partial_results = []
    for chunk in chunks:
        partial_results.append(kmeans_partial(chunk, centroids))

    # NOTE (rerun-simple fix): 'partial_results' e' uma lista de futures
    # (AppFuture) de kmeans_partial ainda nao resolvidos. Passa-la
    # diretamente para kmeans_merge (outra @task) faz o worker tentar
    # indexar/iterar objetos AppFuture em vez de listas reais
    # (TypeError: 'Future' object is not subscriptable, dentro de
    # kmeans_merge). Resolvido com compss_wait_on() antes de repassar a
    # lista para a task de reducao.
    partial_results = compss_wait_on(partial_results)
    # Combina resultados parciais
    merged = kmeans_merge(partial_results)
    final_centroids, counts = compss_wait_on(merged)

    elapsed = time.time() - start
    print("KMeans final centroids:")
    pprint(final_centroids)
    print("Cluster counts:", counts)
    print("Total time: %f s" % elapsed)



def parse_args():
    parser = argparse.ArgumentParser(description="PyCOMPSs workflows: KMeans, Matmul, Lysozyme")
    parser.add_argument("--algorithm", choices=["kmeans", "matmul", "lysozyme"],
                        required=True, help="Algoritmo a executar")
    # Parâmetros comuns
    parser.add_argument("--output", type=str, default=os.getcwd(),
                        help="Diretório de saída (padrão: cwd)")

    # KMeans
    parser.add_argument("--num_points", type=int, default=1000)
    parser.add_argument("--num_clusters", type=int, default=3)
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--num_chunks", type=int, default=4)

    # Matmul
    parser.add_argument("--matrix_size", type=int, default=100)
    parser.add_argument("--block_size", type=int, default=10)

    # Lysozyme
    parser.add_argument("--pdb", type=str, help="Arquivo PDB de entrada")
    parser.add_argument("--force_field", type=str, default="gromos43a1")
    parser.add_argument("--water_model", type=str, default="spc")
    parser.add_argument("--nsteps", type=int, default=100)

    return parser.parse_args()



def main():
    args = parse_args()
    start_time = time.time()

    if args.algorithm == "kmeans":
        main_kmeans(args.num_points, args.num_clusters, args.dim, args.num_chunks)
    elif args.algorithm == "matmul":
        main_matmul(args.matrix_size, args.block_size)
    elif args.algorithm == "lysozyme":
        if not args.pdb:
            raise ValueError("Para Lysozyme, forneça --pdb <arquivo.pdb>")
        main_lysozyme(args.pdb, args.force_field, args.water_model, args.output, args.nsteps)
    else:
        raise ValueError("Algoritmo desconhecido")

    print("Tempo total do workflow: %f s" % (time.time() - start_time))



if __name__ == "__main__":
    main()
