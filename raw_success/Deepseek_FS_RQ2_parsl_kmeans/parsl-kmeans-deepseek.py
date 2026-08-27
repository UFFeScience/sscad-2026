#!/usr/bin/env python3
import os
import parsl
from parsl.app.app import python_app
from parsl.configs.local_threads import config
import numpy as np

def main():
    # Criar diretório de saída
    os.makedirs('output', exist_ok=True)
    
    # Carregar configuração do Parsl
    parsl.load(config)
    
    @python_app
    def kmeans_app(data, n_clusters=3, max_iter=100):
        """
        Implementação simplificada do KMeans usando Parsl
        """
        from sklearn.cluster import KMeans
        import numpy as np
        
        # Converter para numpy array se necessário
        if isinstance(data, list):
            data = np.array(data)
        
        # Executar KMeans
        kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter, random_state=42)
        kmeans.fit(data)
        
        # Retornar resultados
        return {
            'centers': kmeans.cluster_centers_.tolist(),
            'labels': kmeans.labels_.tolist(),
            'inertia': kmeans.inertia_
        }
    
    # Gerar dados de exemplo
    np.random.seed(42)
    X = np.random.randn(100, 2) * 2 + 5  # 100 pontos em 2D
    
    # Executar KMeans
    print("Executando KMeans...")
    result = kmeans_app(X.tolist(), n_clusters=3, max_iter=100)
    result_future = result.result()
    
    print("Resultado do KMeans:")
    print(f"Centros dos clusters: {result_future['centers']}")
    print(f"Rótulos dos clusters: {result_future['labels'][:10]}...")  # Mostrar apenas os primeiros 10
    print(f"Inércia: {result_future['inertia']}")
    
    # Salvar resultados
    import json
    with open('output/kmeans_results.json', 'w') as f:
        json.dump(result_future, f, indent=2)
    print("Resultados salvos em output/kmeans_results.json")
    
    # Limpar Parsl
    parsl.clear()

if __name__ == '__main__':
    main()