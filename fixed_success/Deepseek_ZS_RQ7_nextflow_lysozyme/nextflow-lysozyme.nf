#!/usr/bin/env nextflow

params.config_path   = "/Users/plwms/Documents/dissertacao/testes/automation/fixtures/lysozyme/config"
params.dataset_path  = "/Users/plwms/Documents/dissertacao/testes/automation/fixtures/lysozyme/dataset"
params.output_path   = "results"

params.gmx       = params.gmx       ?: 'gmx'
params.gmx_mpi   = params.gmx_mpi   ?: 'gmx_mpi'
params.mpirun    = params.mpirun    ?: 'mpirun'
params.forcefield = 'oplsaa'
params.water      = 'spce'
params.pname      = 'NA'
params.nname      = 'CL'

process GENERATE_TOPOLOGY {
    input:
    tuple val(id), path(pdb)

    output:
    tuple val(id), path("${id}.gro"),     emit: structure
    tuple val(id), path("${id}.top"),     emit: topology

    script:
    """
    ${params.gmx} pdb2gmx -f $pdb -o ${id}.gro -p ${id}.top -ignh -ff ${params.forcefield} -water ${params.water}
    """
}

process DEFINE_BOX {
    input:
    tuple val(id), path(structure)

    output:
    tuple val(id), path("${id}_newbox.gro"), emit: structure_newbox

    script:
    """
    ${params.gmx} editconf -f $structure -o ${id}_newbox.gro -c -d 1.0 -bt cubic
    """
}

process ADD_SOLVATE {
    input:
    tuple val(id), path(structure_newbox), path(topology_in)

    output:
    tuple val(id), path("${id}_solv.gro"),   emit: protein_solv
    tuple val(id), path("${id}_solv.top"),   emit: topology_out

    script:
    """
    cp $topology_in ${id}_solv.top
    ${params.gmx} solvate -cp $structure_newbox -cs spc216.gro -o ${id}_solv.gro -p ${id}_solv.top
    """
}

process ASSEMBLE_TPR {
    input:
    tuple val(id), path(mdp), path(conf), path(topology), val(stem)

    output:
    tuple val(id), path("${stem}.tpr")

    script:
    """
    ${params.gmx} grompp -f $mdp -c $conf -p $topology -o ${stem}.tpr -maxwarn 2
    """
}

process ASSEMBLE_TPR_EM {
    input:
    tuple val(id), path(mdp), path(conf), path(topology), val(stem)

    output:
    tuple val(id), path("${stem}.tpr")

    script:
    """
    ${params.gmx} grompp -f $mdp -c $conf -p $topology -o ${stem}.tpr -maxwarn 2
    """
}

process REPLACE_SOLVENT_WITH_IONS {
    input:
    tuple val(id), path(ions), path(topology), path(group), val(output_gro), val(output_top)

    output:
    tuple val(id), path(output_gro), emit: protein_solv_ions
    tuple val(id), path(output_top), emit: topology_ions

    script:
    """
    cp $topology $output_top
    ${params.gmx} genion -s $ions -o $output_gro -p $output_top -pname ${params.pname} -nname ${params.nname} -neutral < $group
    """
}

process ENERGY_MINIMIZATION {
    input:
    tuple val(id), path(em_tpr)

    output:
    tuple val(id), path("${id}_em_energy.edr")

    script:
    """
    ${params.mpirun} -np 1 ${params.gmx_mpi} mdrun -v -ntomp 0 -s $em_tpr -e ${id}_em_energy.edr
    """
}

process ENERGY_ANALYSIS {
    publishDir "${params.output_path}", mode: 'copy'

    input:
    tuple val(id), path(em_energy), path(selection), val(output_name)

    output:
    tuple val(id), path(output_name)

    script:
    """
    ${params.gmx} energy -f $em_energy -o $output_name < $selection
    """
}

workflow {
    if (!params.config_path || !params.dataset_path || !params.output_path) {
        throw new Exception("Informe --config_path, --dataset_path e --output_path")
    }

    file(params.output_path).mkdirs()

    def proteins = Channel.fromPath("${params.dataset_path}/*.pdb")
        .map { pdb -> [pdb.baseName, pdb] }

    def topologyResult = GENERATE_TOPOLOGY(proteins)
    def boxResult = DEFINE_BOX(topologyResult.structure)

    def solvatedResult = ADD_SOLVATE(
        boxResult.structure_newbox.join(topologyResult.topology)
    )

    def ionsMdp = file("${params.config_path}/ions.mdp")
    def assembleIonsInput = solvatedResult.protein_solv
        .join(solvatedResult.topology_out)
        .map { id, solv, top -> [id, ionsMdp, solv, top, "${id}_ions"] }

    def ionsTprResult = ASSEMBLE_TPR(assembleIonsInput)

    def genionGroup = file("${params.config_path}/genion.group")
    def replaceInput = ionsTprResult
        .join(solvatedResult.topology_out)
        .map { id, tpr, top -> [id, tpr, top, genionGroup, "${id}_solv_ions.gro", "${id}_solv_ions.top"] }

    def solvIonsResult = REPLACE_SOLVENT_WITH_IONS(replaceInput)

    def minimMdp = file("${params.config_path}/minim.mdp")
    def assembleEmInput = solvIonsResult.protein_solv_ions
        .join(solvIonsResult.topology_ions)
        .map { id, solv_ions, top -> [id, minimMdp, solv_ions, top, "${id}_em"] }

    def emTprResult = ASSEMBLE_TPR_EM(assembleEmInput)
    def emEnergyResult = ENERGY_MINIMIZATION(emTprResult)

    def energySelection = file("${params.config_path}/energy.selection")
    def energyInput = emEnergyResult
        .map { id, energy -> [id, energy, energySelection, "${id}_potential.xvg"] }

    ENERGY_ANALYSIS(energyInput)
}