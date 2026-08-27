nextflow.enable.dsl = 2

/*
 * Parameters equivalent to the PyCOMPSs blocked matrix multiplication
 */
params.num_blocks       = 1
params.elems_per_block  = 2
params.seed             = 0
params.outdir           = 'results_matmul'

process GENERATE_BLOCK {
    tag "${matrix}_${i}_${j}"

    input:
    tuple val(matrix), val(i), val(j), val(size), val(num_blocks), val(seed), val(set_to_zero)

    output:
    tuple val(matrix), val(i), val(j), path("${matrix}_${i}_${j}.npy")

    script:
    """
    python - <<'PY'
import numpy as np

matrix = "${matrix}"
i = int(${i})
j = int(${j})
size = int(${size})
num_blocks = int(${num_blocks})
seed = int(${seed})
set_to_zero = "${set_to_zero}" == "true"

np.random.seed(seed)

if not set_to_zero:
    b = np.random.random((size, size))
    b /= np.sum(b) * float(num_blocks)
else:
    b = np.zeros((size, size))

np.save(f"{matrix}_{i}_{j}.npy", b)
PY
    """
}

process PARTIAL_PRODUCT {
    tag "C_${i}_${j}_from_${k}"

    input:
    tuple val(i), val(j), val(k), path(a_block), path(b_block)

    output:
    tuple val(i), val(j), path("partial_${i}_${j}_${k}.npy")

    script:
    """
    python - <<'PY'
import numpy as np

A = np.load("${a_block}")
B = np.load("${b_block}")

C = np.dot(A, B)

np.save("partial_${i}_${j}_${k}.npy", C)
PY
    """
}

process REDUCE_BLOCK {
    tag "C_${i}_${j}"
    publishDir params.outdir, mode: 'copy'

    input:
    tuple val(i), val(j), path(partials)

    output:
    tuple val(i), val(j), path("C_${i}_${j}.npy")

    script:
    def partial_names = partials.collect { it.name }.join(' ')

    """
    python - <<'PY'
import numpy as np

files = "${partial_names}".split()

accum = None

for f in files:
    block = np.load(f)

    if accum is None:
        accum = block.copy()
    else:
        accum += block

np.save("C_${i}_${j}.npy", accum)
PY
    """
}

process ASSEMBLE_MATRIX {
    tag "assemble_C"
    publishDir params.outdir, mode: 'copy'

    input:
    path blocks
    val num_blocks
    val elems_per_block

    output:
    path "C.npy"
    path "C.txt"

    script:
    def block_names = blocks.collect { it.name }.join(' ')

    """
    python - <<'PY'
import numpy as np
import re

files = "${block_names}".split()

num_blocks = int(${num_blocks})
elems_per_block = int(${elems_per_block})

C = np.zeros(
    (
        num_blocks * elems_per_block,
        num_blocks * elems_per_block
    )
)

pattern = re.compile(r"C_(\\d+)_(\\d+)\\.npy")

for f in files:
    match = pattern.match(f)
    if not match:
        continue

    i = int(match.group(1))
    j = int(match.group(2))

    block = np.load(f)

    r0 = i * elems_per_block
    r1 = r0 + elems_per_block
    c0 = j * elems_per_block
    c1 = c0 + elems_per_block

    C[r0:r1, c0:c1] = block

np.save("C.npy", C)

with open("C.txt", "w") as out:
    out.write(str(C))
    out.write("\\n")
PY
    """
}

workflow {
    log.info """
    B L O C K E D   M A T M U L   N E X T F L O W
    ==============================================
    num_blocks      : ${params.num_blocks}
    elems_per_block : ${params.elems_per_block}
    seed            : ${params.seed}
    outdir          : ${params.outdir}
    """.stripIndent()

    def block_meta = []

    (0..<params.num_blocks).each { i ->
        (0..<params.num_blocks).each { j ->
            /*
             * Equivalent seed logic to the PyCOMPSs code.
             * In the original code, bid is reset for each row.
             */
            def seed_a = params.seed + 2 * j
            def seed_b = params.seed + 2 * j + 1

            block_meta << tuple(
                'A',
                i,
                j,
                params.elems_per_block,
                params.num_blocks,
                seed_a,
                false
            )

            block_meta << tuple(
                'B',
                i,
                j,
                params.elems_per_block,
                params.num_blocks,
                seed_b,
                false
            )
        }
    }

    blocks_ch = GENERATE_BLOCK(Channel.fromList(block_meta))

    /*
     * A blocks keyed by k:
     * A[i][k] -> tuple(k, i, file)
     */
    a_ch = blocks_ch
        .filter { matrix, i, j, file -> matrix == 'A' }
        .map { matrix, i, k, file -> tuple(k, i, file) }

    /*
     * B blocks keyed by k:
     * B[k][j] -> tuple(k, j, file)
     */
    b_ch = blocks_ch
        .filter { matrix, i, j, file -> matrix == 'B' }
        .map { matrix, k, j, file -> tuple(k, j, file) }

    /*
     * Join A[i][k] with B[k][j] for each k.
     */
    products_input_ch = a_ch
        .combine(b_ch, by: 0)
        .map { k, i, a_file, j, b_file ->
            tuple(i, j, k, a_file, b_file)
        }

    partials_ch = PARTIAL_PRODUCT(products_input_ch)

    c_blocks_ch = REDUCE_BLOCK(
        partials_ch.groupTuple(by: [0, 1])
    )

    ASSEMBLE_MATRIX(
        c_blocks_ch.map { i, j, block -> block }.collect(),
        params.num_blocks,
        params.elems_per_block
    )
}