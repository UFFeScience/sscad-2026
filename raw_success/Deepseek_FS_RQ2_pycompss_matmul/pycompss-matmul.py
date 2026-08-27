import os
import time
import argparse
import numpy as np
from pprint import pprint
from pycompss.api.task import task
from pycompss.api.api import compss_wait_on
from pycompss.api.parameter import FILE_IN, FILE_OUT

# ============================================================================
# MATMUL IMPLEMENTATION
# ============================================================================

@task(returns=list)
def generate_matrix(size):
    """Generate a random matrix."""
    return np.random.rand(size, size).tolist()

@task(returns=list)
def matrix_multiply(A_chunk, B):
    """Multiply a chunk of A with matrix B."""
    A_chunk = np.array(A_chunk)
    B = np.array(B)
    return np.dot(A_chunk, B).tolist()

@task(returns=list)
def merge_matrix_chunks(chunk1, chunk2):
    """Merge two matrix chunks vertically."""
    if not chunk1:
        return chunk2
    return chunk1 + chunk2

def matmul_pycompss(size):
    """Matrix multiplication using PyCOMPSs."""
    # Generate matrices A and B
    A = generate_matrix(size)
    B = generate_matrix(size)
    A = compss_wait_on(A)
    B = compss_wait_on(B)

    # Split A into chunks
    chunk_size = max(1, size // 4)
    chunks = [A[i:i+chunk_size] for i in range(0, size, chunk_size)]

    # Parallel matrix multiplication
    result_chunks = []
    for chunk in chunks:
        result_chunks.append(matrix_multiply(chunk, B))

    # Merge results
    final_result = []
    for chunk in result_chunks:
        chunk_result = compss_wait_on(chunk)
        final_result.extend(chunk_result)

    return np.array(final_result)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main(size):
    """Main function to execute the Matrix Multiplication algorithm."""
    start_time = time.time()

    print(f"Running Matrix Multiplication (size {size}x{size})...")
    result = matmul_pycompss(size)
    result = compss_wait_on(result)

    print("Matrix Multiplication Results:")
    print(f"Result matrix shape: {result.shape}")
    print(f"Result matrix (first 5x5): {result[:5, :5]}")

    total_time = time.time() - start_time
    print(f"Total execution time: {total_time:.2f} seconds")

def parse_args():
    """Argument parser."""
    parser = argparse.ArgumentParser(description='PyCOMPSs Matmul Workflow')
    parser.add_argument(
        '-s', '--size',
        type=int,
        default=10,
        help='Matrix size for matmul algorithm (default: 10)'
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args.size)
