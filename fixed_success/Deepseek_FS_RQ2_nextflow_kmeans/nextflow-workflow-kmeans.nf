/*
 * pipeline input parameters
 */
params.algorithm = 'kmeans'
params.k = 3
params.outdir = 'results'

params.kmeans_data = "/Users/plwms/Documents/dissertacao/testes/RQ2/Deepseek/ZS/kmeans_data.txt"
params.matrix_a = "$projectDir/data/matrix_a.txt"
params.matrix_b = "$projectDir/data/matrix_b.txt"
params.pdb = "$projectDir/data/lysozyme.pdb"
params.mdp = "$projectDir/data/em.mdp"

/*
 * KMeans
 */


process KMEANS {
    tag "KMeans"
    cpus 2
    publishDir params.outdir, mode: 'copy'

    input:
    path data
    val k

    output:
    path 'kmeans_labels.txt'
    path 'kmeans_centers.txt'

    script:
    """
    python - <<'PY'
    import numpy as np
    from sklearn.cluster import KMeans

    X = np.loadtxt('$data')
    km = KMeans(n_clusters=$k, random_state=0).fit(X)

    np.savetxt('kmeans_labels.txt', km.labels_, fmt='%d')
    np.savetxt('kmeans_centers.txt', km.cluster_centers_)
    PY
    """
}


workflow KMEANS_WF {
    KMEANS(params.kmeans_data, params.k)
}


workflow {
    KMEANS_WF()
}
