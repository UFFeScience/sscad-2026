#!/usr/bin/env nextflow
nextflow.enable.dsl=2

// Step 1 — Generate topology from PDB file.
process generate_topology {
    tag "$name"

    input:
        tuple val(name), path(pdb)
    output:
        tuple val(name), path("${name}.gro"), path("${name}.top")
    script:
    """
    gmx pdb2gmx \
        -f ${pdb} \
        -o ${name}.gro \
        -p ${name}.top \
        -ignh \
        -ff oplsaa \
        -water spce
    """
}

// Step 2 — Define simulation box.
process define_box {
    tag "$name"

    input:
        tuple val(name), path(gro), path(top)
    output:
        tuple val(name), path("${name}_newbox.gro"), path(top)
    script:
    """
    gmx editconf \
        -f ${gro} \
        -o ${name}_newbox.gro \
        -c \
        -d 1.0 \
        -bt cubic
    """
}

// Step 3 — Add solvent.
process add_solvate {
    tag "$name"

    input:
        tuple val(name), path(newbox), path(top)
    output:
        tuple val(name), path("${name}_solv.gro"), path(top)
    script:
    """
    gmx solvate \
        -cp ${newbox} \
        -cs spc216.gro \
        -o ${name}_solv.gro \
        -p ${top}
    """
}

// Step 4 — Add ions (two grompp + genion calls chained inside one process).
process add_ions {
    tag "$name"

    input:
        tuple val(name), path(solv), path(top)
        path ions_mdp
    output:
        tuple val(name), path("${name}_solv_ions.gro"), path(top)
    script:
    """
    gmx grompp \
        -f ${ions_mdp} \
        -c ${solv} \
        -p ${top} \
        -o ions.tpr

    echo "13" | gmx genion \
        -s ions.tpr \
        -o ${name}_solv_ions.gro \
        -p ${top} \
        -pname NA \
        -nname CL \
        -neutral
    """
}

// Step 5 — Energy minimization (grompp + mdrun).
process energy_minimization {
    tag "$name"

    input:
        tuple val(name), path(solv_ions), path(top)
        path minim_mdp
    output:
        tuple val(name), path("${name}_em.edr")
    script:
    """
    gmx grompp \
        -f ${minim_mdp} \
        -c ${solv_ions} \
        -p ${top} \
        -o em.tpr

    gmx mdrun \
        -v \
        -s em.tpr \
        -e ${name}_em.edr
    """
}

// Step 6 — Energy analysis; extracts potential energy to an XVG file.
process energy_analysis {
    tag "$name"
    publishDir params.outdir, mode: 'copy'

    input:
        tuple val(name), path(edr)
    output:
        path "${name}_potential.xvg"
    script:
    """
    printf "Potential\\n0\\n" | gmx energy \
        -f ${edr} \
        -o ${name}_potential.xvg
    """
}

workflow {
    ions_mdp  = file("${params.config}/ions.mdp")
    minim_mdp = file("${params.config}/minim.mdp")

    // One channel element per PDB file — each protein runs its own pipeline in parallel
    pdbs = Channel
        .fromPath("${params.dataset}/*.pdb")
        .map { f -> tuple(f.baseName, f) }

    topology  = generate_topology(pdbs)
    boxed     = define_box(topology)
    solvated  = add_solvate(boxed)
    ions      = add_ions(solvated, ions_mdp)
    minimized = energy_minimization(ions, minim_mdp)
    energy_analysis(minimized)
}
