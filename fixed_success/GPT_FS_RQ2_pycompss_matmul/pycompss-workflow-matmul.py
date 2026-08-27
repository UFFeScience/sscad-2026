import os



import shlex



import time



import shutil



import argparse



import subprocess



import urllib.request



import numpy as np



from pycompss.api.task import task



from pycompss.api.api import compss_wait_on



from pycompss.api.parameter import FILE_IN, FILE_OUT



def reduce_tree(items, reducer):
    """
    Redução em árvore para evitar uma cadeia longa de dependências.
    """
    if not items:
        return None

    while len(items) > 1:
        new_items = []
        for i in range(0, len(items), 2):
            if i + 1 < len(items):
                new_items.append(reducer(items[i], items[i + 1]))
            else:
                new_items.append(items[i])
        items = new_items

    return items[0]



def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path



@task(returns=list)
def generate_matrix_block(seed, block_size):
    """
    Gera um bloco quadrado da matriz.
    """
    rng = np.random.default_rng(seed)
    block = rng.random((block_size, block_size), dtype=np.float64)
    return block.tolist()



@task(returns=list)
def block_multiply(a, b):
    """
    Multiplica dois blocos.
    """
    a_np = np.asarray(a, dtype=np.float64)
    b_np = np.asarray(b, dtype=np.float64)
    c_np = a_np @ b_np
    return c_np.tolist()



@task(returns=list, priority=True)
def block_add(a, b):
    """
    Soma dois blocos.
    """
    a_np = np.asarray(a, dtype=np.float64)
    b_np = np.asarray(b, dtype=np.float64)
    return (a_np + b_np).tolist()



@task(returns=float)
def block_checksum(block):
    """
    Soma todos os elementos de um bloco.
    """
    return float(np.asarray(block, dtype=np.float64).sum())



@task(returns=float, priority=True)
def add_float(a, b):
    return float(a + b)



def workflow_matmul(args):
    start = time.time()

    if args.size % args.block_size != 0:
        raise ValueError("size deve ser múltiplo de block-size")

    nblocks = args.size // args.block_size

    print("Matriz: {}x{}".format(args.size, args.size))
    print("Blocos: {}x{} de tamanho {}".format(nblocks, nblocks, args.block_size))

    A = []
    B = []

    for i in range(nblocks):
        row_a = []
        row_b = []
        for j in range(nblocks):
            row_a.append(generate_matrix_block(args.seed + 1000 * i + j, args.block_size))
            row_b.append(generate_matrix_block(args.seed + 999999 + 1000 * i + j, args.block_size))
        A.append(row_a)
        B.append(row_b)

    C = []

    for i in range(nblocks):
        row_c = []
        for j in range(nblocks):
            products = []
            for k in range(nblocks):
                products.append(block_multiply(A[i][k], B[k][j]))

            cij = reduce_tree(products, block_add)
            row_c.append(cij)

        C.append(row_c)

    checksums = []
    for i in range(nblocks):
        for j in range(nblocks):
            checksums.append(block_checksum(C[i][j]))

    checksum_future = reduce_tree(checksums, add_float)
    checksum = compss_wait_on(checksum_future)

    elapsed = time.time() - start

    print("\nResultado Matmul")
    print("Checksum de C: {:.10f}".format(checksum))
    print("Tempo total: {:.6f} s".format(elapsed))



def parse_args():
    parser = argparse.ArgumentParser(
        description="Workflows PyCOMPSs: KMeans, Matmul e Lysozyme/GROMACS"
    )

    subparsers = parser.add_subparsers(dest="workflow", required=True)

    # -------------------------
    # KMeans
    # -------------------------
    # KMeans e Lysozyme nao estao implementados neste arquivo (cada
    # algoritmo foi gerado em um arquivo pycompss-workflow-<algo>.py
    # separado) -- os subparsers correspondentes foram removidos daqui
    # porque referenciavam funcoes (workflow_kmeans/workflow_lysozyme)
    # inexistentes neste modulo, causando NameError em parse_args()
    # mesmo ao rodar apenas o subcomando "matmul".

    p_matmul = subparsers.add_parser("matmul", help="Executa workflow Matmul blocado")
    p_matmul.add_argument("--size", type=int, default=1024)
    p_matmul.add_argument("--block-size", type=int, default=256)
    p_matmul.add_argument("--seed", type=int, default=12345)
    p_matmul.set_defaults(func=workflow_matmul)

    return parser.parse_args()



def main():
    args = parse_args()
    args.func(args)



if __name__ == "__main__":
    main()
