params.dataset_path = "$projectDir/data"
params.config_path = "$projectDir/config"
params.outdir = "results"

process GENERATE_TOPOLOGY {
    publishDir params.outdir, mode: 'copy'

    input:
    tuple val(name), path(pdb)

    output:
    tuple val(name), path("${name}.gro"), path("${name}.top")

    script:
    """
    gmx pdb2gmx -f $pdb -o ${name}.gro -p ${name}.top -ignh -ff oplsaa -water spce
    """
}

process DEFINE_BOX {
    publishDir params.outdir, mode: 'copy'

    input:
    tuple val(name), path(structure), path(topology)

    output:
    tuple val(name), path("${name}_newbox.gro"), path(topology)

    script:
    """
    gmx editconf -f $structure -o ${name}_newbox.gro -c -d 1.0 -bt cubic
    """
}

process ADD_SOLVATE {
    publishDir params.outdir, mode: 'copy'

    input:
    tuple val(name), path(structure_newbox), path(topology)

    output:
    tuple val(name), path("${name}_solv.gro"), path(topology)

    script:
    """
    gmx solvate -cp $structure_newbox -cs spc216.gro -o ${name}_solv.gro -p $topology
    """
}

process ASSEMBLE_TPR {
    publishDir params.outdir, mode: 'copy'

    input:
    tuple val(name), path(protein_solv), path(topology)
    path mdp_file

    output:
    tuple val(name), path("${name}.tpr"), path(topology)

    script:
    """
    gmx grompp -f $mdp_file -c $protein_solv -p $topology -o ${name}.tpr
    """
}

process REPLACE_IONS {
    publishDir params.outdir, mode: 'copy'

    input:
    tuple val(name), path(tpr), path(topology)
    path group_file

    output:
    tuple val(name), path("${name}_solv_ions.gro"), path(topology)

    script:
    """
    gmx genion -s $tpr -o ${name}_solv_ions.gro -p $topology -pname NA -nname CL -neutral < $group_file
    """
}

process ENERGY_MINIMIZATION {
    cpus 4
    publishDir params.outdir, mode: 'copy'

    input:
    tuple val(name), path(protein_solv_ions), path(topology)
    path minim_mdp

    output:
    tuple val(name), path("${name}_em.edr")

    script:
    """
    # Primeiro assemble o tpr de minimização
    gmx grompp -f $minim_mdp -c $protein_solv_ions -p $topology -o ${name}_em.tpr
    # Executa a minimização
    gmx mdrun -v -s ${name}_em.tpr
    """
}

process ENERGY_ANALYSIS {
    publishDir params.outdir, mode: 'copy'

    input:
    tuple val(name), path(em_energy)
    path selection_file

    output:
    path "${name}_potential.xvg"

    script:
    """
    gmx energy -f $em_energy -o ${name}_potential.xvg < $selection_file
    """
}

workflow {
    // Cria canal com os arquivos PDB do dataset
    protein_ch = Channel.fromPath("${params.dataset_path}/*.pdb")
        .map { file -> [file.baseName, file] }

    // Fluxo de execução
    topologies = GENERATE_TOPOLOGY(protein_ch)
    boxes = DEFINE_BOX(topologies)
    solvated = ADD_SOLVATE(boxes)
    
    ions_tpr = ASSEMBLE_TPR(solvated, file("${params.config_path}/ions.mdp"))
    ions_gro = REPLACE_IONS(ions_tpr, file("${params.config_path}/genion.group"))
    
    em_results = ENERGY_MINIMIZATION(ions_gro, file("${params.config_path}/minim.mdp"))
    ENERGY_ANALYSIS(em_results, file("${params.config_path}/energy.selection"))
}