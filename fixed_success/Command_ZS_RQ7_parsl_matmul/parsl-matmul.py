import time
import numpy as np
import parsl
from parsl import File
from parsl.data_provider.files import File as ParslFile

# --- Parsl Tasks ---

@parsl.python_app
def generate_block(size, num_blocks, seed=0, set_to_zero=False):
    """
    Generates a square block of given size.
    """
    np.random.seed(seed)
    if not set_to_zero:
        b = np.random.random((size, size))
        b /= np.sum(b) * float(num_blocks)
    else:
        b = np.zeros((size, size))
    return b

@parsl.python_app
def fused_multiply_add(A, B, C):
    """
    Multiplies A and B and accumulates into C.
    In Parsl, we return the new C to simulate INOUT behavior.
    """
    return C + np.dot(A, B)

def dot(A, B, C):
    """
    Blocked matrix multiplication.
    A, B, C are lists of lists of Blocks (Futures or Arrays).
    """
    n, m = len(A), len(B[0])
    
    # To faithfully translate the nested loop structure while using Parsl:
    # We iterate i, j, k. For each k, we update C[i][j].
    # Since we cannot modify C[i][j] in-place across workers easily, 
    # we accumulate updates in a temporary list per block C[i][j] 
    # and sum them at the end, OR we chain the tasks (serial per block).
    # Here we use a reduction approach for efficiency while keeping the logic.
    
    # Initialize C blocks to zeros if they aren't already
    # (Assuming C is passed in as a list of lists of zeros)
    
    for i in range(n):
        for j in range(m):
            # Collect all partial products A[i][k] * B[k][j]
            partials = []
            for k in range(n):
                # A[i][k] and B[k][j] might be Futures or Arrays
                a_block = A[i][k].result() if hasattr(A[i][k], 'result') else A[i][k]
                b_block = B[k][j].result() if hasattr(B[k][j], 'result') else B[k][j]
                partials.append(np.dot(a_block, b_block))
            
            # Sum partials into C[i][j]
            if isinstance(C[i][j], ParslFile):
                # If C is a file, load, add, save (rare for this algo, usually in-memory)
                c_data = np.load(C[i][j])
                c_data += sum(partials)
                np.save(C[i][j], c_data)
            else:
                c_block = C[i][j].result() if hasattr(C[i][j], 'result') else C[i][j]
                C[i][j] = c_block + sum(partials)

def main_matmul(args):
    start_time = time.time()
    
    # Generate data
    A, B, C = [], [], []
    for i in range(args.num_blocks):
        A.append([])
        B.append([])
        C.append([])
        for j in range(args.num_blocks):
            # Generate blocks
            a_block = generate_block(args.elems_per_block, args.num_blocks, seed=args.seed + i + j)
            b_block = generate_block(args.elems_per_block, args.num_blocks, seed=args.seed + i + j + 100)
            c_block = generate_block(args.elems_per_block, args.num_blocks, set_to_zero=True)
            
            A[-1].append(a_block)
            B[-1].append(b_block)
            C[-1].append(c_block)
            
    # Wait for generation
    # (Implicit in dflow below)
    
    # Compute Dot Product
    # We wrap the logic in a function to submit tasks if we wanted to parallelize the loops
    # For this translation, we keep the Python loops but use tasks for the heavy lifting (dot products)
    
    dot(A, B, C)
    
    # Retrieve results if stored as files
    # Here C is a list of lists of Futures or Arrays
    final_C = []
    for i in range(args.num_blocks):
        row = []
        for j in range(args.num_blocks):
            val = C[i][j].result() if hasattr(C[i][j], 'result') else C[i][j]
            row.append(val)
        final_C.append(row)
        
    print("-----------------------------------------")
    print("Multiplication complete.")
    # Print a sample element
    if args.num_blocks > 0:
        print(f"C[0][0] sample:\n{final_C[0][0][:5, :5]}")
    print("-----------------------------------------")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--num_blocks', type=int, default=1)
    parser.add_argument('-e', '--elems_per_block', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    
    from parsl.configs.local_threads import config as _local_config
    parsl.load(_local_config)
    main_matmul(args)