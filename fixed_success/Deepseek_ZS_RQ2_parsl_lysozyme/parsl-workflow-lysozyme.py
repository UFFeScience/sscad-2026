"""
Exemplo de uso do Parsl para criar workflows de:
- KMeans (clustering)
- Matmul (multiplicação de matrizes)
- Lysozyme (simulação molecular com GROMACS)

Requer: parsl, numpy, scikit-learn, e GROMACS instalados (gmx e gmx_mpi).
"""



import os



import numpy as np



import parsl



from parsl import python_app, bash_app



from parsl.config import Config



from parsl.executors import ThreadPoolExecutor



config = Config(
    executors=[ThreadPoolExecutor(max_threads=4)]
)



parsl.load(config)



@bash_app
def pdb2gmx_app(pdb_file, output_dir, force_field="gromos53a6", water_model="spc"):
    """Executa o pdb2gmx para gerar topologia e arquivos de estrutura."""
    cmd = f"gmx pdb2gmx -f {pdb_file} -o {output_dir}/conf.gro -p {output_dir}/topol.top -ff {force_field} -water {water_model}"
    return cmd



@bash_app
def grompp_app(conf_file, topol_file, mdp_file, output_tpr):
    """Executa o grompp para gerar o arquivo .tpr."""
    cmd = f"gmx grompp -f {mdp_file} -c {conf_file} -p {topol_file} -o {output_tpr} -maxwarn 2"
    return cmd



@bash_app
def mdrun_app(tpr_file, output_prefix, mpi=False):
    """Executa o mdrun (minimização ou dinâmica) usando gmx ou gmx_mpi."""
    if mpi:
        cmd = f"gmx_mpi mdrun -s {tpr_file} -deffnm {output_prefix}"
    else:
        cmd = f"gmx mdrun -s {tpr_file} -deffnm {output_prefix}"
    return cmd



def run_lysozyme_workflow(pdb_file, mdp_file, output_dir, use_mpi=True):
    """
    Orquestra o workflow do Lysozyme:
    1. pdb2gmx (gera topologia)
    2. grompp (prepara o sistema)
    3. mdrun (executa a simulação)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Passo 1: pdb2gmx
    conf_future = pdb2gmx_app(pdb_file, output_dir)
    conf_future.result()  # bash_apps abaixo usam strings de path, nao Futures,
    # entao o Parsl nao ve dependencia de dados e lancava as 3 etapas quase em
    # paralelo (grompp lendo conf.gro/topol.top antes de existirem) -- espera
    # explicita para serializar
    conf_file = f"{output_dir}/conf.gro"
    topol_file = f"{output_dir}/topol.top"

    # Passo 2: grompp (depende do passo 1)
    tpr_file = f"{output_dir}/sim.tpr"
    grompp_future = grompp_app(conf_file, topol_file, mdp_file, tpr_file)
    grompp_future.result()  # idem: mdrun precisa que sim.tpr ja exista

    # Passo 3: mdrun (depende do passo 2)
    output_prefix = f"{output_dir}/sim"
    mdrun_future = mdrun_app(tpr_file, output_prefix, mpi=use_mpi)

    # Aguarda a conclusão e retorna o caminho do resultado
    mdrun_future.result()
    print(f"Lysozyme simulation completed. Outputs in {output_dir}")
    return output_prefix



if __name__ == "__main__":
    # Executa apenas o workflow Lysozyme (kmeans/matmul nao estao definidos
    # neste arquivo -- cada algoritmo foi gerado em um arquivo separado)
    print("\n=== Workflow Lysozyme ===")
    pdb_file = "lysozyme.pdb"          # Estrutura inicial da lisozima
    mdp_file = "minim.mdp"             # Arquivo de parâmetros de simulação
    output_dir = "lysozyme_output"
    run_lysozyme_workflow(pdb_file, mdp_file, output_dir, use_mpi=True)
