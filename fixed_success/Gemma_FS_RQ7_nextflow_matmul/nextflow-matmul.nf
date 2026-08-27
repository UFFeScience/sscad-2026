params.num_blocks = 1
params.elems_per_block = 2
params.seed = 0
params.outdir = "results_matmul"

process GENERATE_BLOCKS {
    output:
    path "blocks_A/*.npy", emit: a
    path "blocks_B/*.npy", emit: b
    path "blocks_C/*.npy", emit: c

    script:
    """
    python3 -c "
import numpy as np
import os
for folder in ['blocks_A', 'blocks_B', 'blocks_C']: os.makedirs(folder, exist_ok=True)
for i in range(${params.num_blocks}):
    for j in range(${params.num_blocks}):
        # Bloco A
        np.random.seed(${params.seed} + (i*${params.num_blocks} + j))
        b_a = np.random.random((${params.elems_per_block}, ${params.elems_per_block}))
        np.save(f'blocks_A/A_{i}_{j}.npy', b_a / (np.sum(b_a) * ${params.num_blocks}))
        # Bloco B
        np.random.seed(${params.seed} + (i*${params.num_blocks} + j) + 1)
        b_b = np.random.random((${params.elems_per_block}, ${params.elems_per_block}))
        np.save(f'blocks_B/B_{i}_{j}.npy', b_b / (np.sum(b_b) * ${params.num_blocks}))
        # Bloco C (Zero)
        np.save(f'blocks_C/C_{i}_{j}.npy', np.zeros((${params.elems_per_block}, ${params.elems_per_block})))
"
    """
}

process FUSED_MULTIPLY_ADD {
    input:
    path block_a
    path block_b
    path block_c

    output:
    path "res_c.npy"

    script:
    """
    python3 -c "
import numpy as np
a = np.load('${block_a}')
b = np.load('${block_b}')
c = np.load('${block_c}')
res = c + np.dot(a, b)
np.save('res_c.npy', res)
"
    """
}

workflow {
    blocks = GENERATE_BLOCKS()
    // Aqui o Nextflow distribuiria as combinações de blocos i,j,k conforme a lógica do dot()
    FUSED_MULTIPLY_ADD(blocks.a, blocks.b, blocks.c)
}