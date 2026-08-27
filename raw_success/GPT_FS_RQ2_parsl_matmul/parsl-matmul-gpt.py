#!/usr/bin/env python3

import os
import parsl

from parsl.app.app import python_app
from parsl.configs.local_threads import config

os.makedirs("output", exist_ok=True)

parsl.load(config)


@python_app
def generate_matrix(size):

    import numpy as np

    np.random.seed()

    return np.random.rand(size, size)


@python_app
def multiply(A, B):

    import numpy as np

    return np.matmul(A, B)


@python_app
def save_matrix(C):

    import numpy as np

    np.save("output/matmul.npy", C)

    return C.shape


if __name__ == "__main__":

    A = generate_matrix(2000)

    B = generate_matrix(2000)

    C = multiply(A, B)

    shape = save_matrix(C)

    print("Resultado:", shape.result())

    parsl.clear()
