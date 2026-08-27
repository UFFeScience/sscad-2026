/*
 * pipeline input parameters
 */
params.seed = 0
params.numpoints = 100
params.dimensions = 2
params.num_centres = 5
params.fragments = 10
params.mode = "uniform"
params.iterations = 20
params.epsilon = 1e-9
params.arity = 50
params.outdir = "results"

/*
 * Gera um fragmento de dados (arquivo .npy)
 */
process GENERATE_FRAGMENT {
    tag "Fragment ${offset}"
    publishDir "$params.outdir/fragments", mode: 'copy'

    input:
    val seed
    val numpoints
    val dimensions
    val mode
    val offset

    output:
    path "fragment_${offset}.npy"

    script:
    """
    python - <<CODE
    import numpy as np
    points_per_fragment = max(1, ${numpoints} // ${params.fragments})
    start = ${offset} * points_per_fragment
    end = min(${numpoints}, start + points_per_fragment)
    npoints = end - start
    np.random.seed(${seed} + start)
    r = np.random.normal if '${mode}'=='normal' else np.random.random
    mat = np.asarray([r(${dimensions}) for _ in range(npoints)])
    mat -= np.min(mat)
    mx = np.max(mat)
    if mx > 0.0:
        mat /= mx
    np.save('fragment_${offset}.npy', mat)
    CODE
    """
}

/*
 * Executa o algoritmo K-Means sobre os fragmentos
 */
process KMEANS {
    publishDir "$params.outdir", mode: 'copy'

    input:
    path fragments
    val dimensions
    val num_centres
    val iterations
    val seed
    val epsilon
    val arity

    output:
    path "centres.npy"

    script:
    """
    python - <<CODE
    import numpy as np
    from sklearn.metrics import pairwise_distances
    from sklearn.metrics.pairwise import paired_distances

    # Carregar fragmentos
    fragment_files = sorted('${fragments}'.split())
    fragments = [np.load(f) for f in fragment_files]

    # Inicializar centros
    np.random.seed(${seed})
    centres = np.asarray([np.random.random(${dimensions}) for _ in range(${num_centres})])
    old_centres = None
    iteration = 0

    def converged(old, new, eps, it, max_it):
        if old is None:
            return False
        dist = np.sum(paired_distances(new, old))
        return dist < eps**2 or it >= max_it

    while not converged(old_centres, centres, ${epsilon}, iteration, ${iterations}):
        old_centres = centres.copy()
        partials = []
        for frag in fragments:
            close = pairwise_distances(frag, old_centres).argmin(axis=1)
            partial = np.zeros((${num_centres}, 2), dtype=object)
            for c in range(${num_centres}):
                idx = np.argwhere(close == c).flatten()
                partial[c][0] = np.sum(frag[idx], axis=0)
                partial[c][1] = idx.shape[0]
            partials.append(partial)

        # Redução (merge) dos parciais
        while len(partials) > 1:
            subset = partials[:${arity}]
            partials = partials[${arity}:]
            acc = subset[0].copy()
            for d in subset[1:]:
                acc = acc + d
            partials.append(acc)

        # Recalcular centros
        for idx, sum_ in enumerate(partials[0]):
            if sum_[1] != 0:
                centres[idx] = sum_[0] / sum_[1]
        iteration += 1

    np.save('centres.npy', centres)
    CODE
    """
}

workflow {
    log.info """\
        K M E A N S - N F   P I P E L I N E
        ===================================
        numpoints   : ${params.numpoints}
        dimensions  : ${params.dimensions}
        num_centres : ${params.num_centres}
        fragments   : ${params.fragments}
        iterations  : ${params.iterations}
        outdir      : ${params.outdir}
        """.stripIndent(true)

    // Gera os fragmentos
    offsets_ch = Channel.of(0..params.fragments-1)
    frag_ch = GENERATE_FRAGMENT(params.seed, params.numpoints, params.dimensions, params.mode, offsets_ch)

    // Coleta todos os fragmentos em um único diretório
    all_fragments = frag_ch.collect()

    // Executa K-Means
    KMEANS(all_fragments, params.dimensions, params.num_centres, params.iterations, params.seed, params.epsilon, params.arity)
}