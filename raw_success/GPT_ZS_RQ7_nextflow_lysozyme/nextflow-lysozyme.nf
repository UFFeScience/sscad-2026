nextflow.enable.dsl = 2

params.config  = null
params.dataset = null
params.outdir  = 'results_lysozyme'

params.gmx     = System.getenv('GMX_BIN') ? "${System.getenv('GMX_BIN')}/gmx" : 'gmx'
params.gmx_mpi = 'gmx_mpi'

params.cpus_mdrun = 4


process GENERATE_TOPOLOGY {

    tag "${name}"

    publishDir params.outdir, mode: 'copy'

    input:
    tuple val(name), path(protein)

    output:
    tuple val(name), path("${name}.gro"), path("${name}.top")

    script:
    """
    ${params.gmx} pdb2gmx \\
        -f ${protein} \\
        -o ${name}.gro \\
        -p ${name}.top \\
        -ignh \\
        -ff oplsaa \\
        -water spce
    """
}


process DEFINE_BOX {

    tag "${name}"

    publishDir params.outdir, mode: 'copy'

    input:
    tuple val(name), path(structure), path(topology)

    output:
    tuple val(name), path("${name}_newbox.gro"), path(topology)

    script:
    """
    ${params.gmx} editconf \\
        -f ${structure} \\
        -o ${name}_newbox.gro \\
        -c \\
        -d 1.0 \\
        -bt cubic
    """
}


process ADD_SOLVATE {

    tag "${name}"

    publishDir params.outdir, mode: 'copy'

    input:
    tuple val(name), path(structure_newbox), path(topology)

    output:
    tuple val(name), path("${name}_solv.gro"), path("${name}_solv.top")

    script:
    """
    cp ${topology} ${name}_solv.top

    ${params.gmx} solvate \\
        -cp ${structure_newbox} \\
        -cs spc216.gro \\
        -o ${name}_solv.gro \\
        -p ${name}_solv.top
    """
}


process ASSEMBLE_IONS_TPR {

    tag "${name}"

    publishDir params.outdir, mode: 'copy'

    input:
    tuple val(name), path(protein_solv), path(topology)

    output:
    tuple val(name), path("${name}_ions.tpr"), path(protein_solv), path(topology)

    script:
    """
    ${params.gmx} grompp \\
        -f ${params.config}/ions.mdp \\
        -c ${protein_solv} \\
        -p ${topology} \\
        -o ${name}_ions.tpr
    """
}


process REPLACE_SOLVENT_WITH_IONS {

    tag "${name}"

    publishDir params.outdir, mode: 'copy'

    input:
    tuple val(name), path(ions), path(protein_solv), path(topology)

    output:
    tuple val(name), path("${name}_solv_ions.gro"), path("${name}_ions.top")

    script:
    """
    cp ${topology} ${name}_ions.top

    ${params.gmx} genion \\
        -s ${ions} \\
        -o ${name}_solv_ions.gro \\
        -p ${name}_ions.top \\
        -pname NA \\
        -nname CL \\
        -neutral \\
        < ${params.config}/genion.group
    """
}


process ASSEMBLE_EM_TPR {

    tag "${name}"

    publishDir params.outdir, mode: 'copy'

    input:
    tuple val(name), path(protein_solv_ions), path(topology)

    output:
    tuple val(name), path("${name}_em.tpr")

    script:
    """
    ${params.gmx} grompp \\
        -f ${params.config}/minim.mdp \\
        -c ${protein_solv_ions} \\
        -p ${topology} \\
        -o ${name}_em.tpr
    """
}


process ENERGY_MINIMIZATION {

    tag "${name}"

    cpus params.cpus_mdrun

    publishDir params.outdir, mode: 'copy'

    input:
    tuple val(name), path(em)

    output:
    tuple val(name), path("${name}_em_energy.edr")

    script:
    """
    mpirun -np ${task.cpus} ${params.gmx_mpi} mdrun \\
        -v \\
        -ntomp 0 \\
        -s ${em} \\
        -e ${name}_em_energy.edr
    """
}


process ENERGY_ANALYSIS {

    tag "${name}"

    publishDir params.outdir, mode: 'copy'

    input:
    tuple val(name), path(em_energy)

    output:
    path "${name}_potential.xvg"

    script:
    """
    ${params.gmx} energy \\
        -f ${em_energy} \\
        -o ${name}_potential.xvg \\
        < ${params.config}/energy.selection
    """
}


workflow {

    if (!params.config) {
        error "Informe --config com o diretório contendo ions.mdp, minim.mdp, genion.group e energy.selection"
    }

    if (!params.dataset) {
        error "Informe --dataset com o diretório contendo os arquivos .pdb"
    }

    proteins = Channel
        .fromPath("${params.dataset}/*")
        .filter { it.isFile() }
        .map { pdb ->
            tuple(pdb.baseName, pdb)
        }

    topology_ch = GENERATE_TOPOLOGY(proteins)

    newbox_ch = DEFINE_BOX(topology_ch)

    solv_ch = ADD_SOLVATE(newbox_ch)

    ions_tpr_ch = ASSEMBLE_IONS_TPR(solv_ch)

    solv_ions_ch = REPLACE_SOLVENT_WITH_IONS(ions_tpr_ch)

    em_tpr_ch = ASSEMBLE_EM_TPR(solv_ions_ch)

    em_energy_ch = ENERGY_MINIMIZATION(em_tpr_ch)

    ENERGY_ANALYSIS(em_energy_ch)
}