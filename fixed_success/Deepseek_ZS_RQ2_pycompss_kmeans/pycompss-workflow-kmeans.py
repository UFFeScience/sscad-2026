"""
PyCOMPSs: KMeans, Matmul e Lysozyme (GROMACS)

Executar com:
    runcompss --lang=python --python_interpreter=python3 workflow_pycompss.py
"""

import pathlib
import subprocess
import tempfile
import numpy as np
from pycompss.api.task import task
from pycompss.api.parameter import FILE_IN, FILE_OUT, DIRECTORY_INOUT
from pycompss.api.api import compss_wait_on

BASE_DIR = pathlib.Path("/tmp/pycompss_workflows")

LYSOZYME_PDB = BASE_DIR / "1AKI.pdb"

MPI_RANKS = 4


@task(dataset=FILE_IN, returns=1)
def kmeans_task(dataset, k=3, max_iter=200, seed=42):
    from sklearn.cluster import KMeans

    X = np.load(dataset)
    model = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=10,
        max_iter=max_iter,
        random_state=seed,
    )
    model.fit(X)

    return {
        "centroids": model.cluster_centers_.tolist(),
        "labels": model.labels_.tolist(),
        "inertia": float(model.inertia_),
    }


def workflow_kmeans(dataset_path):
    return kmeans_task(
        dataset=str(dataset_path),
        k=3,
        max_iter=200,
        seed=42,
    )


def _create_inputs():
    BASE_DIR.mkdir(parents=True, exist_ok=True)

    # Dados para KMeans
    dataset = BASE_DIR / "kmeans_dataset.npy"
    rng = np.random.default_rng(42)
    if not dataset.exists():
        X = np.concatenate([
            rng.normal(loc=[0, 0], scale=0.5, size=(30, 2)),
            rng.normal(loc=[5, 5], scale=0.5, size=(30, 2)),
        ])
        np.save(dataset, X)

    return dataset


def main():
    # NOTE (rerun-simple fix): o arquivo original era um extrato de um script
    # combinado (kmeans+matmul+lysozyme); o main() original ainda chamava
    # workflow_matmul()/workflow_lysozyme(), que não estão definidos neste
    # arquivo (NameError), e ainda exigia um PDB de lisozima que este
    # workflow de KMeans nem usa. Corrigido para só executar o workflow
    # realmente definido aqui (KMeans).
    dataset = _create_inputs()

    kmeans_future = workflow_kmeans(dataset)
    kmeans_result = compss_wait_on(kmeans_future)

    print("KMeans:", kmeans_result)


if __name__ == "__main__":
    main()
