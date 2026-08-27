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


def _ensure_parent(path):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)


@task(a_path=FILE_IN, b_path=FILE_IN, c_path=FILE_OUT, returns=1)
def matmul_task(a_path, b_path, c_path):
    A = np.load(a_path)
    B = np.load(b_path)
    C = A @ B

    _ensure_parent(c_path)
    np.save(c_path, C)

    return {
        "shape_a": A.shape,
        "shape_b": B.shape,
        "shape_c": C.shape,
    }


def workflow_matmul(a_path, b_path, c_path):
    return matmul_task(
        a_path=str(a_path),
        b_path=str(b_path),
        c_path=str(c_path),
    )


def _create_inputs():
    BASE_DIR.mkdir(parents=True, exist_ok=True)

    # Matrizes para Matmul
    a = BASE_DIR / "A.npy"
    b = BASE_DIR / "B.npy"
    rng = np.random.default_rng(42)
    if not a.exists():
        np.save(a, rng.random((64, 64)))
    if not b.exists():
        np.save(b, rng.random((64, 64)))

    return a, b


def main():
    # NOTE (rerun-simple fix): o arquivo original era um extrato de um script
    # combinado (kmeans+matmul+lysozyme); o main() original ainda chamava
    # workflow_kmeans()/workflow_lysozyme(), que não estão definidos neste
    # arquivo (NameError), e ainda exigia um PDB de lisozima que este
    # workflow de Matmul nem usa. Corrigido para só executar o workflow
    # realmente definido aqui (Matmul).
    a, b = _create_inputs()

    c_path = BASE_DIR / "C_result.npy"
    c_path.unlink(missing_ok=True)

    matmul_future = workflow_matmul(a, b, c_path)
    matmul_result = compss_wait_on(matmul_future)

    print("Matmul:", matmul_result)


if __name__ == "__main__":
    main()
