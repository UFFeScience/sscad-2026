#!/usr/bin/env python3
import numpy as np
import sys

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 run_matmul.py <matrix_a.txt> <matrix_b.txt>")
        sys.exit(1)

    file_a = sys.argv[1]
    file_b = sys.argv[2]

    try:
        # Load matrices from text files
        # Assumes space or tab separated values
        mat_a = np.loadtxt(file_a)
        mat_b = np.loadtxt(file_b)

        # Perform Matrix Multiplication
        result = np.matmul(mat_a, mat_b)

        # Print result to stdout (Nextflow captures this via '>')
        print(result)

    except Exception as e:
        print(f"Error during matrix multiplication: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()