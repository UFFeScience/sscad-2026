import numpy as np
import time

from pycompss.api.task import task
from pycompss.api.api import compss_wait_on
from pycompss.api.api import compss_barrier
from pycompss.api.parameter import IN, OUT


# ============================================================
# LYSOZYME WORKFLOW
# ============================================================


@task(returns=float)
def calculate_energy(atom_coordinates):

    """
    Simulação simples de energia molecular.
    """

    energy = np.sum(
        atom_coordinates ** 2
    )

    return energy



@task(returns=np.ndarray)
def optimize_structure(coords, energy):

    """
    Atualiza estrutura molecular.
    """

    return coords - 0.01 * energy



def lysozyme_workflow(
        atoms,
        iterations=10):


    print("\nExecutando Lysozyme")


    coordinates = atoms


    for _ in range(iterations):

        energy = calculate_energy(
            coordinates
        )


        coordinates = optimize_structure(
            coordinates,
            energy
        )


        coordinates = compss_wait_on(
            coordinates
        )


    print("Lysozyme finalizado")


    return coordinates


# ============================================================
# MAIN WORKFLOW
# ============================================================


def main():

    # -------------------------------
    # Lysozyme
    # -------------------------------


    atoms = np.random.rand(
        200,
        3
    )


    structure = lysozyme_workflow(
        atoms
    )


    print(
        "Estrutura final:",
        structure.shape
    )


    compss_barrier()


if __name__ == "__main__":

    main()
