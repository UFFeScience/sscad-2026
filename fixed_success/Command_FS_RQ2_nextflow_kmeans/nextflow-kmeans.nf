/*
 * -----------------------------------------------------------------
 *  Parâmetros do pipeline KMeans
 * -----------------------------------------------------------------
 */
params.kmeans = [:]
params.kmeans.input   = "/Users/plwms/Documents/dissertacao/testes/RQ2/Command/C-EXEMPLOS/data.txt"
params.kmeans.n_clusters = 3
params.outdir = "results/kmeans"

/*
 * -----------------------------------------------------------------
 *  Processo KMeans – executa scikit‑learn via Python
 * -----------------------------------------------------------------
 */
process KMEANS {
    cpus 1
    memory '1 GB'

    input:
    path data_file

    output:
    path 'kmeans_result.txt'   // arquivo simples com uma etiqueta por linha

    script:
    """
    # Carrega os dados, executa KMeans e grava os rótulos
    python3 - <<'PY'
import numpy as np
from sklearn.cluster import KMeans

# lê uma matriz CSV (sem cabeçalho, uma observação por linha)
X = np.loadtxt('$data_file', delimiter=',')
kmeans = KMeans(n_clusters=${params.kmeans.n_clusters}, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)

# grava os rótulos – um por linha
np.savetxt('kmeans_result.txt', labels, fmt='%d')
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
        K M E A N S   N F   P I P E L I N E
        ======================================
        input file  : ${params.kmeans.input}
        n clusters  : ${params.kmeans.n_clusters}
        outdir      : ${params.outdir}
        """.stripIndent(true)

    // Canal que fornece o arquivo de dados
    channel
        .fromPath(params.kmeans.input)
        .set { data_ch }

    // Executa o agrupamento
    KMEANS(data_ch)

    // Opcional: imprime o resultado na saída padrão quando o pipeline termina
    workflow.onComplete {
        println "KMeans finalizado – rótulos escritos em ${params.outdir}/kmeans_result.txt"
    }
}