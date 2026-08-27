/*
 * -----------------------------------------------------------------
 *  Parâmetros do pipeline Matmul
 * -----------------------------------------------------------------
 */
params.matmul = [:]
params.matmul.A = "/Users/plwms/Documents/dissertacao/testes/RQ2/Command/C-EXEMPLOS/A.txt"
params.matmul.B = "/Users/plwms/Documents/dissertacao/testes/RQ2/Command/C-EXEMPLOS/B.txt"
params.outdir = "results/matmul"

/*
 * -----------------------------------------------------------------
 *  Processo Matmul – multiplicação simples usando NumPy
 * -----------------------------------------------------------------
 */
process MATMUL {
    cpus 1
    memory '2 GB'

    input:
    path A
    path B

    output:
    path 'C.txt'   // matriz resultante, um elemento por linha

    script:
    """
    # Lê duas matrizes (formato simples: uma linha por linha, valores separados por espaço)
    # e escreve o produto C = A × B
    python3 - <<'PY'
import numpy as np

A = np.loadtxt('$A', delimiter=',')
B = np.loadtxt('$B', delimiter=',')
C = A @ B                     # operador de multiplicação de matrizes do NumPy
np.savetxt('C.txt', C)
PY
    """
}

/*
 * -----------------------------------------------------------------
 *  Fluxo de trabalho
 * -----------------------------------------------------------------
 */
workflow {
    log.info """\
        M A T M U L   N F   P I P E L I N E
        ======================================
        matrix A   : ${params.matmul.A}
        matrix B   : ${params.matmul.B}
        outdir     : ${params.outdir}
        """.stripIndent(true)

    // Fornece os dois arquivos como canais separados
    a_ch = channel.fromPath(params.matmul.A)
    b_ch = channel.fromPath(params.matmul.B)

    MATMUL(a_ch, b_ch)

    workflow.onComplete {
        println "Matmul finalizado – resultado escrito em ${params.outdir}/C.txt"
    }
}