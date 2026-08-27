#!/usr/bin/env python3
"""
Script Parsl para execução do workflow:
- KMeans (scikit-learn)
"""

import parsl
from parsl import python_app, bash_app, File
from parsl.config import Config
from parsl.executors import ThreadPoolExecutor, HighThroughputExecutor
from parsl.providers import LocalProvider
import os
import shutil
import tempfile
import logging

# Configuração do logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURAÇÃO DO Parsl
# ============================================================================

config = Config(
    executors=[
        ThreadPoolExecutor(
            label='threads',
            max_threads=4
        ),
        HighThroughputExecutor(
            label='htex',
            provider=LocalProvider(
                init_blocks=1,
                max_blocks=1,
            ),
            max_workers_per_node=4,
        )
    ],
    strategy='none',
)

parsl.load(config)

# ============================================================================
# WORKFLOW: KMEANS CLUSTERING
# ============================================================================

@python_app
def kmeans_app(n_samples=1000, n_features=10, n_clusters=3, random_state=42):
    """
    App Python para executar KMeans clustering
    """
    import numpy as np
    from sklearn.cluster import KMeans
    import logging

    logger = logging.getLogger(__name__)

    np.random.seed(random_state)
    X = np.random.randn(n_samples, n_features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X)

    result = {
        'n_clusters': n_clusters,
        'inertia': float(kmeans.inertia_),
        'n_iter': kmeans.n_iter_,
        'cluster_centers_shape': kmeans.cluster_centers_.shape
    }

    logger.info(f"KMeans completado: {result}")
    return result

def workflow_kmeans():
    """Executa workflow KMeans"""
    logger.info("=" * 60)
    logger.info("INICIANDO WORKFLOW KMEANS")
    logger.info("=" * 60)

    futures = []
    for i in range(3):
        n_clusters = 3 + i
        future = kmeans_app(
            n_samples=1000,
            n_features=10,
            n_clusters=n_clusters,
            random_state=42 + i
        )
        futures.append(future)

    results = [f.result() for f in futures]

    logger.info("Resultados KMeans:")
    for i, r in enumerate(results):
        logger.info(f"  Execução {i+1}: {r}")

    return results

# ============================================================================
# FUNÇÃO PRINCIPAL
# ============================================================================

def main():
    """
    Função principal para executar o workflow KMeans
    """
    logger.info("=" * 60)
    logger.info("INICIANDO EXECUÇÃO DO WORKFLOW")
    logger.info("=" * 60)

    try:
        kmeans_results = workflow_kmeans()

        logger.info("\n" + "=" * 60)
        logger.info("RESUMO FINAL")
        logger.info("=" * 60)
        logger.info(f"KMeans: {len(kmeans_results)} execuções completadas")

    except Exception as e:
        logger.error(f"Erro durante execução: {e}")
        raise

    finally:
        parsl.dfk().cleanup()
        logger.info("Parsl finalizado")

if __name__ == "__main__":
    main()
