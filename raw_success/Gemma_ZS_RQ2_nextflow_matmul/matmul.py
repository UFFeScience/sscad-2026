import sys
import numpy as np

def run_matmul(file1, file2):
    # Load matrices from text files
    m1 = np.loadtxt(file1)
    m2 = np.loadtxt(file2)
    
    # Perform multiplication
    result = np.dot(m1, m2)
    
    # Print to stdout (which Nextflow redirects to result_matrix.txt)
    print(result)

if __name__ == "__main__":
    run_matmul(sys.argv[1], sys.argv[2])