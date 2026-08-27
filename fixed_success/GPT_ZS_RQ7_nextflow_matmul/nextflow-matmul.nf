nextflow.enable.dsl = 2

params.num_blocks      = 1
params.elems_per_block = 2
params.seed            = 0
params.outdir          = 'results_matmul'


process GENERATE_A_BLOCK {

    tag "A[${i},${k}]"

    publishDir params.outdir, mode: 'copy'

    input:
    tuple val(i), val(k), val(block_size), val(num_blocks), val(seed)

    output:
    tuple val(k), val(i), path("A_${i}_${k}.npy")

    script:
    """
    python3 - <<'PY'
import numpy as np

i          = ${i}
k          = ${k}
size       = ${block_size}
num_blocks = ${num_blocks}
seed       = ${seed}

np.random.seed(seed)

b = np.random.random((size, size))
b /= np.sum(b) * float(num_blocks)

np.save(f"A_{i}_{k}.npy", b)
PY
    """
}


process GENERATE_B_BLOCK {

    tag "B[${k},${j}]"

    publishDir params.outdir, mode: 'copy'

    input:
    tuple val(k), val(j), val(block_size), val(num_blocks), val(seed)

    output:
    tuple val(k), val(j), path("B_${k}_${j}.npy")

    script:
    """
    python3 - <<'PY'
import numpy as np

k          = ${k}
j          = ${j}
size       = ${block_size}
num_blocks = ${num_blocks}
seed       = ${seed}

np.random.seed(seed)

b = np.random.random((size, size))
b /= np.sum(b) * float(num_blocks)

np.save(f"B_{k}_{j}.npy", b)
PY
    """
}


process BLOCK_PRODUCT {

    tag "P[${i},${j},${k}]"

    publishDir params.outdir, mode: 'copy'

    input:
    tuple val(i), val(j), val(k), path(A), path(B)

    output:
    tuple val(i), val(j), path("P_${i}_${j}_${k}.npy")

    script:
    """
    python3 - <<'PY'
import numpy as np

i = ${i}
j = ${j}
k = ${k}

A = np.load("${A}")
B = np.load("${B}")

P = np.dot(A, B)

np.save(f"P_{i}_{j}_{k}.npy", P)
PY
    """
}


process SUM_BLOCK {

    tag "C[${i},${j}]"

    publishDir params.outdir, mode: 'copy'

    input:
    tuple val(i), val(j), path(partials)

    output:
    path "C_${i}_${j}.npy"

    script:
    """
    python3 - <<'PY'
import glob
import numpy as np

i = ${i}
j = ${j}

partial_files = sorted(glob.glob(f"P_{i}_{j}_*.npy"))

acc = None

for pf in partial_files:
    block = np.load(pf)

    if acc is None:
        acc = np.zeros_like(block)

    acc += block

np.save(f"C_{i}_{j}.npy", acc)
PY
    """
}


workflow {

    num_blocks      = params.num_blocks as int
    elems_per_block = params.elems_per_block as int
    base_seed       = params.seed as int

    a_infos = []
    b_infos = []

    /*
     * O código PyCOMPSs reinicializa bid para cada linha i.
     * Portanto, A[i][j] usa seed + 2*j
     * e B[i][j] usa seed + 2*j + 1.
     */
    (0..<num_blocks).each { i ->
        (0..<num_blocks).each { k ->
            a_infos << tuple(
                i,
                k,
                elems_per_block,
                num_blocks,
                base_seed + 2 * k
            )
        }
    }

    (0..<num_blocks).each { k ->
        (0..<num_blocks).each { j ->
            b_infos << tuple(
                k,
                j,
                elems_per_block,
                num_blocks,
                base_seed + 2 * j + 1
            )
        }
    }

    A_blocks = GENERATE_A_BLOCK(Channel.fromList(a_infos))
    B_blocks = GENERATE_B_BLOCK(Channel.fromList(b_infos))

    /*
     * Junta A[i,k] com B[k,j] usando o índice k.
     */
    products_input = A_blocks
        .combine(B_blocks, by: 0)
        .map { k, i, Afile, j, Bfile ->
            tuple(i, j, k, Afile, Bfile)
        }

    partial_products = BLOCK_PRODUCT(products_input)

    /*
     * Agrupa todos os P[i,j,k] para o mesmo par i,j.
     * Depois reduz para C[i,j].
     */
    grouped_products = partial_products.groupTuple(
        by: [0, 1],
        size: num_blocks
    )

    SUM_BLOCK(grouped_products)
}