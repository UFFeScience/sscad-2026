import os
import sys

import parsl

from parsl import bash_app
from parsl.data_provider.files import File


from parsl.config import Config
from parsl.executors import HighThroughputExecutor



parsl.load(
    Config(
        executors=[
            HighThroughputExecutor(
                label="hpc"
            )
        ]
    )
)



# -----------------
# GROMACS tasks
# -----------------


@bash_app
def generate_topology(
        protein,
        structure,
        topology,
        stdout=None,
        stderr=None):


    return f"""

    gmx pdb2gmx \
    -f {protein} \
    -o {structure} \
    -p {topology} \
    -ignh \
    -ff oplsaa \
    -water spce

    """



@bash_app
def define_box(
        structure,
        output,
        stdout=None,
        stderr=None):


    return f"""

    gmx editconf \
    -f {structure} \
    -o {output} \
    -c -d 1.0 \
    -bt cubic

    """



@bash_app
def add_solvate(
        structure,
        topology,
        output,
        stdout=None,
        stderr=None):


    return f"""

    gmx solvate \
    -cp {structure} \
    -cs spc216.gro \
    -o {output} \
    -p {topology}

    """



@bash_app
def assemble_tpr(
        conf,
        structure,
        topology,
        output,
        stdout=None,
        stderr=None):


    return f"""

    gmx grompp \
    -f {conf} \
    -c {structure} \
    -p {topology} \
    -o {output}

    """



@bash_app
def replace_solvent_with_ions(
        ions,
        topology,
        output,
        group,
        stdout=None,
        stderr=None):


    return f"""

    gmx genion \
    -s {ions} \
    -o {output} \
    -p {topology} \
    -pname NA \
    -nname CL \
    -neutral < {group}

    """



@bash_app
def energy_minimization(
        em,
        energy,
        stdout=None,
        stderr=None):


    return f"""

    gmx_mpi mdrun \
    -v \
    -s {em} \
    -e {energy}

    """



@bash_app
def energy_analysis(
        em,
        output,
        selection,
        stdout=None,
        stderr=None):


    return f"""

    gmx energy \
    -f {em} \
    -o {output} \
    < {selection}

    """



# -----------------
# Workflow
# -----------------


def main(
        dataset,
        output,
        config):


    for pdb in os.listdir(dataset):


        name=pdb.split(".")[0]


        protein=os.path.join(
            dataset,pdb
        )


        structure=os.path.join(
            output,
            name+".gro"
        )


        topology=os.path.join(
            output,
            name+".top"
        )


        generate_topology(
            protein,
            structure,
            topology
        ).result()



        newbox=os.path.join(
            output,
            name+"_newbox.gro"
        )


        define_box(
            structure,
            newbox
        ).result()



        solv=os.path.join(
            output,
            name+"_solv.gro"
        )


        add_solvate(
            newbox,
            topology,
            solv
        ).result()



        ions=os.path.join(
            output,
            name+"_ions.tpr"
        )


        assemble_tpr(
            os.path.join(config,"ions.mdp"),
            solv,
            topology,
            ions
        ).result()



        solvions=os.path.join(
            output,
            name+"_solv_ions.gro"
        )


        replace_solvent_with_ions(
            ions,
            topology,
            solvions,
            os.path.join(
                config,
                "genion.group"
            )
        ).result()



        em=os.path.join(
            output,
            name+"_em.tpr"
        )


        assemble_tpr(
            os.path.join(config,"minim.mdp"),
            solvions,
            topology,
            em
        ).result()



        energy=os.path.join(
            output,
            name+"_em_energy.edr"
        )


        energy_minimization(
            em,
            energy
        ).result()



        energy_analysis(
            energy,
            os.path.join(
                output,
                name+"_potential.xvg"
            ),
            os.path.join(
                config,
                "energy.selection"
            )
        ).result()



if __name__=="__main__":

    main(
        sys.argv[2],
        sys.argv[3],
        sys.argv[1]
    )