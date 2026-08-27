#!/usr/bin/env python3
"""
Workflow Parsl para simulação de Lysozyme usando GROMACS.
Requer GROMACS instalado (gmx e gmx_mpi) e um arquivo PDB de entrada.
"""
import os
import parsl
from parsl.app.app import bash_app
from parsl.data_provider.files import File
from parsl.configs.local_threads import config

parsl.load(config)

# Diretório de trabalho
WORK_DIR = "lysozyme_workflow"
os.makedirs(WORK_DIR, exist_ok=True)

# Arquivos de entrada/saída esperados
INPUT_PDB = os.path.join(WORK_DIR, "1AKI.pdb")  # Exemplo: estrutura da lisozima
TOPOL = os.path.join(WORK_DIR, "topol.top")
GRO = os.path.join(WORK_DIR, "conf.gro")
TPR = os.path.join(WORK_DIR, "em.tpr")
MD_GRO = os.path.join(WORK_DIR, "md.gro")

@bash_app
def pdb2gmx(input_pdb, outputs=[], stdout='pdb2gmx.out', stderr='pdb2gmx.err'):
    """Gera topologia e arquivo .gro a partir do PDB."""
    return f"gmx pdb2gmx -f {input_pdb} -o {outputs[0].filepath} -p {outputs[1].filepath} -ff gromos43a1 -water spc"

@bash_app
def grompp(inputs=[], outputs=[], stdout='grompp.out', stderr='grompp.err'):
    """Gera o arquivo .tpr para o mdrun."""
    # inputs[0] = conf.gro, inputs[1] = topol.top
    return (f"gmx grompp -f /Users/plwms/Documents/dissertacao/testes/automation/fixtures/lysozyme/config/minim.mdp -c {inputs[0].filepath} -p {inputs[1].filepath} "
            f"-o {outputs[0].filepath} -maxwarn 2")

@bash_app
def mdrun(inputs=[], outputs=[], stdout='mdrun.out', stderr='mdrun.err'):
    """Executa a dinâmica molecular usando gmx_mpi."""
    # inputs[0] = topol.tpr
    return f"gmx_mpi mdrun -s {inputs[0].filepath} -o {outputs[0].filepath} -c {outputs[1].filepath}"

def main():
    # Verifica se o arquivo PDB existe
    if not os.path.exists(INPUT_PDB):
        print(f"Erro: arquivo PDB {INPUT_PDB} não encontrado. Coloque a estrutura da lisozima neste caminho.")
        return

    # Etapa 1: pdb2gmx
    pdb2gmx_future = pdb2gmx(
        INPUT_PDB,
        outputs=[File(GRO), File(TOPOL)],
        stdout=os.path.join(WORK_DIR, "pdb2gmx.out"),
        stderr=os.path.join(WORK_DIR, "pdb2gmx.err")
    )

    # Etapa 2: grompp (depende dos arquivos gerados por pdb2gmx)
    grompp_future = grompp(
        inputs=[pdb2gmx_future.outputs[0], pdb2gmx_future.outputs[1]],  # conf.gro, topol.top
        outputs=[File(TPR)],
        stdout=os.path.join(WORK_DIR, "grompp.out"),
        stderr=os.path.join(WORK_DIR, "grompp.err")
    )

    # Etapa 3: mdrun (depende do .tpr)
    mdrun_future = mdrun(
        inputs=[grompp_future.outputs[0]],
        outputs=[File(os.path.join(WORK_DIR, "md.trr")), File(MD_GRO)],
        stdout=os.path.join(WORK_DIR, "mdrun.out"),
        stderr=os.path.join(WORK_DIR, "mdrun.err")
    )

    # Aguarda a conclusão do workflow
    mdrun_future.result()
    print("Simulação concluída. Arquivos gerados em", WORK_DIR)

    parsl.clear()

if __name__ == "__main__":
    main()