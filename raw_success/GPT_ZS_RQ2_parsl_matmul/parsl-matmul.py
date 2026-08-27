import parsl
from parsl.app.app import python_app
from parsl.configs.local_threads import config
import numpy as np


parsl.load(config)



@python_app
def matrix_multiply(A, B):

    import numpy as np

    C = np.matmul(A, B)

    return C



def matmul_workflow():


    size = 2000


    A = np.random.rand(
        size,
        size
    )


    B = np.random.rand(
        size,
        size
    )


    # Workflow assíncrono
    future = matrix_multiply(
        A,
        B
    )


    C = future.result()


    print(
        "Multiplicação finalizada"
    )


    print(
        "Dimensão resultado:",
        C.shape
    )



if __name__ == "__main__":

    matmul_workflow()