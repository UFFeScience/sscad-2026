nextflow.enable.dsl=2


params.pdb = "$projectDir/data/1aki.pdb"
params.config = "$projectDir/config"

params.outdir="results"



process PDB2GMX {


    input:

    path pdb



    output:

    tuple path("protein.gro"),
          path("topol.top")



    script:


    """
    gmx pdb2gmx \
        -f ${pdb} \
        -o protein.gro \
        -p topol.top \
        -ignh \
        -ff oplsaa \
        -water spce
    """

}



process EDITCONF {


    input:

    tuple path(gro),
          path(top)



    output:

    tuple path("box.gro"),
          path(top)



    script:


    """
    gmx editconf \
        -f ${gro} \
        -o box.gro \
        -c \
        -d 1.0 \
        -bt cubic
    """

}



process SOLVATE {


    input:

    tuple path(box),
          path(top)



    output:

    tuple path("solv.gro"),
          path(top)



    script:


    """
    gmx solvate \
        -cp ${box} \
        -cs spc216.gro \
        -o solv.gro \
        -p ${top}
    """

}



process GROMPP_IONS {


    input:

    tuple path(gro),
          path(top)



    output:

    tuple path("ions.tpr"),
          path(gro),
          path(top)



    script:


    """
    gmx grompp \
        -f ${params.config}/ions.mdp \
        -c ${gro} \
        -p ${top} \
        -o ions.tpr
    """

}



process GENION {


    input:

    tuple path(tpr),
          path(gro),
          path(top)



    output:

    tuple path("solv_ions.gro"),
          path(top)



    script:


    """
    gmx genion \
        -s ${tpr} \
        -o solv_ions.gro \
        -p ${top} \
        -pname NA \
        -nname CL \
        -neutral \
        < ${params.config}/genion.group
    """

}



process GROMPP_MIN {


    input:

    tuple path(gro),
          path(top)



    output:

    path "em.tpr"



    script:


    """
    gmx grompp \
        -f ${params.config}/minim.mdp \
        -c ${gro} \
        -p ${top} \
        -o em.tpr
    """

}



process ENERGY_MIN {


    cpus 4


    input:

    path tpr



    output:

    path "em.edr"



    script:


    """
    gmx_mpi mdrun \
        -v \
        -s ${tpr} \
        -e em.edr
    """

}



process ENERGY_ANALYSIS {


    input:

    path edr



    output:

    path "potential.xvg"



    script:


    """
    gmx energy \
        -f ${edr} \
        -o potential.xvg \
        < ${params.config}/energy.selection
    """

}



workflow {


    topology = PDB2GMX(
        params.pdb
    )


    box = EDITCONF(
        topology
    )


    solv = SOLVATE(
        box
    )


    ions = GROMPP_IONS(
        solv
    )


    ionized = GENION(
        ions
    )


    minim = GROMPP_MIN(
        ionized
    )


    energy = ENERGY_MIN(
        minim
    )


    ENERGY_ANALYSIS(
        energy
    )

}