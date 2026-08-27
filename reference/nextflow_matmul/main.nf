#!/usr/bin/env nextflow
nextflow.enable.dsl=2

// Generates one block (tile) of matrix A or B and saves it as a .npy file.
// All blocks are generated in parallel (one process per block).
process generate_block {
    input:
        tuple val(mat), val(row), val(col), val(seed)
    output:
        tuple val(mat), val(row), val(col), path("block_${mat}_${row}_${col}.npy")
    script:
    """
    python3 << 'PYEOF'
import numpy as np
np.random.seed(${seed})
b = np.random.random((${params.elems_per_block}, ${params.elems_per_block}))
b /= b.sum() * float(${params.num_blocks})
np.save("block_${mat}_${row}_${col}.npy", b)
PYEOF
    """
}

// Multiplies block A[i][k] by block B[k][j] and saves the partial product.
// There is one process per (i, j, k) triple — all run in parallel.
process multiply_block {
    input:
        tuple val(i), val(j), val(k), path(block_a), path(block_b)
    output:
        tuple val(i), val(j), path("partial_${i}_${j}_${k}.npy")
    script:
    """
    python3 << 'PYEOF'
import numpy as np
A = np.load("${block_a}")
B = np.load("${block_b}")
np.save("partial_${i}_${j}_${k}.npy", np.dot(A, B))
PYEOF
    """
}

// Sums all partial products for one result cell C[i][j].
// groupTuple(by:[0,1]) delivers all k products for each (i,j) pair together.
process accumulate_block {
    input:
        tuple val(i), val(j), path(partials)
    output:
        tuple val(i), val(j), path("result_${i}_${j}.npy")
    script:
    """
    python3 << 'PYEOF'
import numpy as np, glob
files  = sorted(glob.glob("partial_*.npy"))
result = sum(np.load(f) for f in files)
np.save("result_${i}_${j}.npy", result)
PYEOF
    """
}

// Collects all C[i][j] result blocks and prints a summary.
process report {
    publishDir params.outdir, mode: 'copy'

    input:
        path results
    output:
        path "matrix_C.npy"
    script:
    """
    python3 << 'PYEOF'
import numpy as np, glob
n = ${params.num_blocks}
s = ${params.elems_per_block}
C = np.zeros((n * s, n * s))
for f in glob.glob("result_*.npy"):
    parts = f.replace(".npy", "").split("_")
    i, j = int(parts[1]), int(parts[2])
    C[i*s:(i+1)*s, j*s:(j+1)*s] = np.load(f)
np.save("matrix_C.npy", C)
print("Matrix C shape:", C.shape)
print("C[0][0] (top-left block sum):", C[:s, :s].sum().round(6))
PYEOF
    """
}

workflow {
    n = params.num_blocks

    // Build the list of block generation tasks replicating the original bid counter
    def block_params = []
    (0..<n).each { i ->
        def bid = 0
        (0..<n).each { j ->
            block_params.add(['A', i, j, params.seed + bid]); bid = bid + 1
            block_params.add(['B', i, j, params.seed + bid]); bid = bid + 1
        }
    }

    generated = generate_block(Channel.fromList(block_params))

    // Key A[i][k] by k (A's column index) → (k, i, file)
    a_keyed = generated
        .filter  { mat, row, col, f -> mat == 'A' }
        .map     { mat, row, col, f -> tuple(col, row, f) }

    // Key B[k][j] by k (B's row index) → (k, j, file)
    b_keyed = generated
        .filter  { mat, row, col, f -> mat == 'B' }
        .map     { mat, row, col, f -> tuple(row, col, f) }

    // Cross-join on k → (k, i, fa, j, fb) → rearrange to (i, j, k, fa, fb)
    mult_inputs = a_keyed
        .combine(b_keyed, by: 0)
        .map     { k, i, fa, j, fb -> tuple(i, j, k, fa, fb) }

    products = multiply_block(mult_inputs)

    // Group all k products for each (i,j) cell, then accumulate
    accumulated = accumulate_block(products.groupTuple(by: [0, 1]))

    report(accumulated.map { i, j, f -> f }.collect())
}
