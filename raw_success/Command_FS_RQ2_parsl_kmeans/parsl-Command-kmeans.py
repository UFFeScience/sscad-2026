#!/usr/bin/env python3
"""
kmeans.py
Workflow Parsl para calcular o algoritmo KMeans.
Gera dados aleatórios, executa o clustering e salva os resultados.
"""

import os
import parsl
from parsl.app.app import python_app, bash_app
from parsl.configs.local_threads import config
from parsl.data_provider.files import File
import numpy as np
from sklearn.cluster import KMeans

# ---------------------------------------------------------------------
# Configuração e inicialização
# ---------------------------------------------------------------------
# Criar diretório de saída
os.makedirs('output', exist_ok=True)

def main():
    # Carregar a configuração do Parsl
    parsl.load(config)

    # ---------------------------------------------------------------------
    # Definir a aplicação Python para KMeans
    # ---------------------------------------------------------------------
    @python_app
    def run_kmeans():
        """
        Gera dados aleatórios e executa o KMeans.
        """
        # 1. Gerar dados sintéticos
        # 100 pontos em 2 dimensões
        np.random.seed(42)
        X = np.random.rand(100, 2) * 100

        # 2. Executar KMeans
        # Usando 3 clusters
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        kmeans.fit(X)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_

        # 3. Salvar resultados em um arquivo
        output_path = 'output/kmeans_results.txt'
        with open(output_path, 'w') as f:
            f.write("Centros dos Clusters:\n")
            f.write(np.array2string(centers, precision=2, suppress_small=True) + "\n")
            f.write("\nRótulos dos Pontos:\n")
            f.write(str(labels) + "\n")
            
        print(f"KMeans concluído. Resultados salvos em {output_path}")
        return output_path

    # ---------------------------------------------------------------------
    # Executar o workflow
    # ---------------------------------------------------------------------
    # Submitir a tarefa
    result_future = run_kmeans()
    
    # Aguardar a conclusão
    output_file = result_future.result()
    
    # Imprimir conteúdo do arquivo de resultado
    with open(output_file, 'r') as f:
        print("\n--- Conteúdo do Arquivo de Resultados ---")
        print(f.read())

    # Limpar a infraestrutura do Parsl
    parsl.clear()

if __name__ == "__main__":
    main()