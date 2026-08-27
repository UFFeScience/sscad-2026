params.num_blocks = 1
params.elems_per_block = 2
params.seed = 0
params.outdir = "matmul_results"

process GENERATE_BLOCK {
    input:
    tuple val(block_id), val(size), val(num_blocks), val(seed), val(set_to_zero)

    output:
    path "block_${block_id}.npy"

    script:
    """
    python3 -c "
import numpy as np
np.random.seed(${seed})
if not ${set_to_zero}:
    b = np.random.random((${size}, ${size}))
    b /= (np.sum(b) * float(${num_blocks}))
else:
    b = np.zeros((${size}, ${size}))
np.save('block_${block_id}.npy', b)
"
    """
}

process FUSED_MULTIPLY_ADD {
    input:
    path block_A
    path block_B
    path block_C_old

    output:
    path "block_C_new.npy"

    script:
    """
    python3 -c "
import numpy as np
A = np.load('${block_A}')
B = np.load('${block_B}')
C = np.load('${block_C_old}')
C += np.dot(A, B)
np.save('block_C_new.npy', C)
"
    """
}

workflow {
    // Logic to generate blocks for A, B and C
    // This would involve creating channels for all combinations of i, j, k
    // and piping them into FUSED_MULTIPLY_ADD.
}