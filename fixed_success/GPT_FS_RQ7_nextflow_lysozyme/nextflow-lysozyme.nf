nextflow.enable.dsl = 2

/*
 * Parameters
 */
params.config = 'config'
params.dataset = 'dataset'
params.outdir = 'results_gromacs'

/*
 * Commands can be customized from the command line:
 *
 * nextflow run gromacs.nf --gmx_cmd /path/to/gmx --gmx_mpi_cmd gmx_mpi
 */
params.gmx_cmd = 'gmx'
params.gmx_mpi_cmd = 'gmx_mpi'

params.forcefield = 'oplsaa'
params.water = 'spce'
params.distance = '1.0'
params.boxtype = 'cubic'

params.mdrun_cpus = 4

process GENERATE_TOPOLOGY {
    tag "$name"
    publishDir params.outdir, mode: 'copy'

    input:
    tuple val(name), path(protein)

    output:
    tuple val(name), path("${name}.gro"), path("${name}.top")

    script:
    """
    ${params.gmx_cmd} pdb2gmx \\
        -f ${protein} \\
        -o ${name}.gro \\
        -p ${name}.top \\
        -ignh \\
        -ff ${params.forcefield} \\
        -water ${params.water}
    """
}

process DEFINE_BOX {
    tag "$name"
    publishDir params.outdir, mode: 'copy'

    input:
    tuple val(name), path(structure), path(topology)

    output:
    tuple val(name), path("${name}_newbox.gro"), path("${name}.top")

    script:
    """
    cp ${topology} ${name}.top

    ${params.gmx_cmd} editconf \\
        -f ${structure} \\
        -o ${name}_newbox.gro \\
        -c \\
        -d ${params.distance} \\
        -bt ${params.boxtype}
    """
}

process ADD_SOLVATE {
    tag "$name"
    publishDir params.outdir, mode: 'copy'

    input:
    tuple val(name), path(structure_newbox), path(topology)

    output:
    tuple val(name), path("${name}_solv.gro"), path("${name}_solv.top")

    script:
    """
    cp ${topology} ${name}_solv.top

    ${params.gmx_cmd} solvate \\
        -cp ${structure_newbox} \\
        -cs spc216.gro \\
        -o ${name}_solv.gro \\
        -p ${name}_solv.top
    """
}

process ASSEMBLE_IONS_TPR {
    tag "$name"
    publishDir params.outdir, mode: 'copy'

    input:
    tuple val(name), path(protein_solv), path(topology)
    path ions_mdp

    output:
    tuple val(name), path("${name}_ions.tpr"), path(protein_solv), path(topology)

    script:
    """
    ${params.gmx_cmd} grompp \\
        -f ${ions_mdp} \\
        -c ${protein_solv} \\
        -p ${topology} \\
        -o ${name}_ions.tpr
    """
}

process REPLACE_SOLVENT_WITH_IONS {
    tag "$name"
    publishDir params.outdir, mode: 'copy'

    input:
    tuple val(name), path(ions), path(protein_solv), path(topology)
    path group

    output:
    tuple val(name), path("${name}_solv_ions.gro"), path("${name}_ions.top")

    script:
    """
    cp ${topology} ${name}_ions.top

    ${params.gmx_cmd} genion \\
        -s ${ions} \\
        -o ${name}_solv_ions.gro \\
        -p ${name}_ions.top \\
        -pname NA \\
        -nname CL \\
        -neutral \\
        < ${group}
    """
}

process ASSEMBLE_EM_TPR {
    tag "$name"
    publishDir params.outdir, mode: 'copy'

    input:
    tuple val(name), path(protein_solv_ions), path(topology)
    path minim_mdp

    output:
    tuple val(name), path("${name}_em.tpr")

    script:
    """
    ${params.gmx_cmd} grompp \\
        -f ${minim_mdp} \\
        -c ${protein_solv_ions} \\
        -p ${topology} \\
        -o ${name}_em.tpr
    """
}

process ENERGY_MINIMIZATION {
    tag "$name"
    cpus params.mdrun_cpus
    publishDir params.outdir, mode: 'copy'

    input:
    tuple val(name), path(em)

    output:
    tuple val(name), path("${name}_em_energy.edr")

    script:
    """
    mpirun -np ${task.cpus} ${params.gmx_mpi_cmd} mdrun \\
        -v \\
        -ntomp 0 \\
        -s ${em} \\
        -e ${name}_em_energy.edr
    """
}

process ENERGY_ANALYSIS {
    tag "$name"
    publishDir params.outdir, mode: 'copy'

    input:
    tuple val(name), path(em_energy)
    path selection

    output:
    tuple val(name), path("${name}_potential.xvg")

    script:
    """
    ${params.gmx_cmd} energy \\
        -f ${em_energy} \\
        -o ${name}_potential.xvg \\
        < ${selection}
    """
}

workflow {
    main:

    log.info """
    G R O M A C S   N E X T F L O W
    ===============================
    config path : ${params.config}
    dataset     : ${params.dataset}
    outdir      : ${params.outdir}
    gmx         : ${params.gmx_cmd}
    gmx_mpi     : ${params.gmx_mpi_cmd}
    """.stripIndent()

    proteins_ch = Channel
        .fromPath("${params.dataset}/*", checkIfExists: true)
        .filter { it.isFile() }
        .map { pdb ->
            tuple(pdb.baseName, pdb)
        }

    ions_mdp_ch = Channel.value(file("${params.config}/ions.mdp"))
    minim_mdp_ch = Channel.value(file("${params.config}/minim.mdp"))
    genion_group_ch = Channel.value(file("${params.config}/genion.group"))
    energy_selection_ch = Channel.value(file("${params.config}/energy.selection"))

    topology_ch = GENERATE_TOPOLOGY(proteins_ch)

    newbox_ch = DEFINE_BOX(topology_ch)

    solv_ch = ADD_SOLVATE(newbox_ch)

    ions_tpr_ch = ASSEMBLE_IONS_TPR(
        solv_ch,
        ions_mdp_ch
    )

    solv_ions_ch = REPLACE_SOLVENT_WITH_IONS(
        ions_tpr_ch,
        genion_group_ch
    )

    em_tpr_ch = ASSEMBLE_EM_TPR(
        solv_ions_ch,
        minim_mdp_ch
    )

    em_energy_ch = ENERGY_MINIMIZATION(em_tpr_ch)

    ENERGY_ANALYSIS(
        em_energy_ch,
        energy_selection_ch
    )

    onComplete:
    log.info workflow.success ?
        "Pipeline finalizado com sucesso. Resultados em: ${params.outdir}" :
        "Erro durante a execução do pipeline."
}