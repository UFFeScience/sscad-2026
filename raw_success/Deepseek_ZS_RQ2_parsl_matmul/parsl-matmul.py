#!/usr/bin/env python3
"""
Script Parsl para execução do workflow:
- Matmul (multiplicação de matrizes com NumPy)
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
# WORKFLOW: MATMUL (Multiplicação de Matrizes)
# ============================================================================

@python_app
def matmul_app(size=100):
    """
    App Python para multiplicação de matrizes com NumPy
    """
    import numpy as np
    import logging

    logger = logging.getLogger(__name__)

    np.random.seed(42)
    A = np.random.randn(size, size)
    B = np.random.randn(size, size)

    C = np.matmul(A, B)

    result = {
        'size': size,
        'shape': C.shape,
        'norm': float(np.linalg.norm(C)),
        'mean': float(np.mean(C)),
        'std': float(np.std(C))
    }

    logger.info(f"Matmul completado para size={size}: {result}")
    return result

@python_app
def matmul_optimized_app(size=100, num_threads=4):
    """
    App Python para multiplicação de matrizes otimizada com NumPy
    """
    import numpy as np
    import os
    import logging

    logger = logging.getLogger(__name__)

    os.environ['OMP_NUM_THREADS'] = str(num_threads)

    np.random.seed(42)
    A = np.random.randn(size, size)
    B = np.random.randn(size, size)

    C = np.dot(A, B)

    result = {
        'size': size,
        'shape': C.shape,
        'norm': float(np.linalg.norm(C)),
        'threads': num_threads
    }

    logger.info(f"Matmul otimizado completado para size={size}")
    return result

def workflow_matmul():
    """Executa workflow Matmul com diferentes tamanhos"""
    logger.info("=" * 60)
    logger.info("INICIANDO WORKFLOW MATMUL")
    logger.info("=" * 60)

    sizes = [50, 100, 200, 500]
    futures = []

    for size in sizes:
        future = matmul_app(size=size)
        futures.append(future)

    for size in [100, 200]:
        future = matmul_optimized_app(size=size, num_threads=2)
        futures.append(future)

    results = [f.result() for f in futures]

    logger.info("Resultados Matmul:")
    for i, r in enumerate(results):
        logger.info(f"  Execução {i+1}: {r}")

    return results

# ============================================================================
# FUNÇÃO PRINCIPAL
# ============================================================================

def main():
    """
    Função principal para executar o workflow Matmul
    """
    logger.info("=" * 60)
    logger.info("INICIANDO EXECUÇÃO DO WORKFLOW")
    logger.info("=" * 60)

    try:
        matmul_results = workflow_matmul()

        logger.info("\n" + "=" * 60)
        logger.info("RESUMO FINAL")
        logger.info("=" * 60)
        logger.info(f"Matmul: {len(matmul_results)} execuções completadas")

    except Exception as e:
        logger.error(f"Erro durante execução: {e}")
        raise

    finally:
        parsl.dfk().cleanup()
        logger.info("Parsl finalizado")

if __name__ == "__main__":
    main()
