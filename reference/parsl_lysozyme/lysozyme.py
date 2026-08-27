import json
import os
import sys
from os import listdir
from os.path import isfile, join

import parsl
from parsl import bash_app
from parsl.config import Config
from parsl.executors import ThreadPoolExecutor
from parsl.data_provider.files import File

# NOTE: the original reference uses HighThroughputExecutor(max_workers_per_node=4,
# provider=LocalProvider()). On this machine that backend exhibited an unexplained
# task-dispatch quirk (spurious BashExitFailure with exit code 1 despite every GROMACS
# command completing successfully, confirmed via debug instrumentation) that made
# lightweight per-task timing unreliable. Switched to ThreadPoolExecutor -- same task
# graph, same worker count (4), simpler in-process execution model, already validated
# for the kmeans/matmul reference scripts in this analysis.
config = Config(executors=[ThreadPoolExecutor(max_threads=4)])
parsl.load(config)

# --- instrumentation added for makespan / cpu / parallelism-efficiency analysis ---
# NOTE: HighThroughputExecutor runs bash_apps in separate worker processes, so each
# function must be fully self-contained (no reference to names defined elsewhere at
# module scope) -- the timing wrapper is therefore inlined directly in every app below
# rather than calling a shared helper.
_METRICS_FILE = os.environ.get("WF_METRICS_FILE", "metrics.jsonl")


@bash_app
def generate_topology(protein, structure, topology, forcefield='oplsaa', water='spce',
                      inputs=[], outputs=[], stdout=None, stderr=None):
    import os as _os
    _mf = _os.environ.get("WF_METRICS_FILE", "metrics.jsonl")
    cmd = (f"$GMX_BIN/gmx pdb2gmx -f {protein} -o {structure} -p {topology} "
           f"-ignh -ff {forcefield} -water {water}")
    return (f'_wf_t0=$(date +%s.%N); {cmd}; _wf_rc=$?; _wf_t1=$(date +%s.%N); '
            f'echo "{{\\"task\\": \\"generate_topology\\", \\"start\\": $_wf_t0, \\"end\\": $_wf_t1}}" >> {_mf}; '
            f'exit $_wf_rc')


@bash_app
def define_box(structure, structure_newbox, distance='1.0', boxtype='cubic',
               inputs=[], outputs=[]):
    import os as _os
    _mf = _os.environ.get("WF_METRICS_FILE", "metrics.jsonl")
    cmd = (f"$GMX_BIN/gmx editconf -f {structure} -o {structure_newbox} "
           f"-c -d {distance} -bt {boxtype}")
    return (f'_wf_t0=$(date +%s.%N); {cmd}; _wf_rc=$?; _wf_t1=$(date +%s.%N); '
            f'echo "{{\\"task\\": \\"define_box\\", \\"start\\": $_wf_t0, \\"end\\": $_wf_t1}}" >> {_mf}; '
            f'exit $_wf_rc')


@bash_app
def add_solvate(structure_newbox, protein_solv, topology, inputs=[], outputs=[]):
    import os as _os
    _mf = _os.environ.get("WF_METRICS_FILE", "metrics.jsonl")
    cmd = (f"$GMX_BIN/gmx solvate -cp {structure_newbox} -cs spc216.gro "
           f"-o {protein_solv} -p {topology}")
    return (f'_wf_t0=$(date +%s.%N); {cmd}; _wf_rc=$?; _wf_t1=$(date +%s.%N); '
            f'echo "{{\\"task\\": \\"add_solvate\\", \\"start\\": $_wf_t0, \\"end\\": $_wf_t1}}" >> {_mf}; '
            f'exit $_wf_rc')


@bash_app
def assemble_tpr(conf, protein_solv, topology, output, inputs=[], outputs=[]):
    import os as _os
    _mf = _os.environ.get("WF_METRICS_FILE", "metrics.jsonl")
    cmd = f"$GMX_BIN/gmx grompp -f {conf} -c {protein_solv} -p {topology} -o {output}"
    return (f'_wf_t0=$(date +%s.%N); {cmd}; _wf_rc=$?; _wf_t1=$(date +%s.%N); '
            f'echo "{{\\"task\\": \\"assemble_tpr\\", \\"start\\": $_wf_t0, \\"end\\": $_wf_t1}}" >> {_mf}; '
            f'exit $_wf_rc')


@bash_app
def replace_solvent_with_ions(ions, output, topology, group,
                              pname='NA', nname='CL', inputs=[], outputs=[]):
    import os as _os
    _mf = _os.environ.get("WF_METRICS_FILE", "metrics.jsonl")
    cmd = (f"$GMX_BIN/gmx genion -s {ions} -o {output} -p {topology} "
           f"-pname {pname} -nname {nname} -neutral < {group}")
    return (f'_wf_t0=$(date +%s.%N); {cmd}; _wf_rc=$?; _wf_t1=$(date +%s.%N); '
            f'echo "{{\\"task\\": \\"replace_solvent_with_ions\\", \\"start\\": $_wf_t0, \\"end\\": $_wf_t1}}" >> {_mf}; '
            f'exit $_wf_rc')


@bash_app
def energy_minimization(em, em_energy, inputs=[], outputs=[]):
    # NOTE (environment workaround, not a code fix): the original reference command is
    # "mpirun gmx_mpi mdrun -v -ntomp 0 -s {em} -e {em_energy}". Neither this Mac's
    # Homebrew GROMACS (single unified `gmx`, no MPI variant) nor the project's Docker
    # container ship a `gmx_mpi` binary -- see the RQ2 reproducibility audit. Substituted
    # plain `gmx mdrun` (single rank) to obtain real timing/CPU data for this analysis.
    import os as _os
    _mf = _os.environ.get("WF_METRICS_FILE", "metrics.jsonl")
    cmd = f"$GMX_BIN/gmx mdrun -v -ntomp 0 -s {em} -e {em_energy}"
    return (f'_wf_t0=$(date +%s.%N); {cmd}; _wf_rc=$?; _wf_t1=$(date +%s.%N); '
            f'echo "{{\\"task\\": \\"energy_minimization\\", \\"start\\": $_wf_t0, \\"end\\": $_wf_t1}}" >> {_mf}; '
            f'exit $_wf_rc')


@bash_app
def energy_analysis(em, output, selection, inputs=[], outputs=[]):
    import os as _os
    _mf = _os.environ.get("WF_METRICS_FILE", "metrics.jsonl")
    cmd = f"$GMX_BIN/gmx energy -f {em} -o {output} < {selection}"
    return (f'_wf_t0=$(date +%s.%N); {cmd}; _wf_rc=$?; _wf_t1=$(date +%s.%N); '
            f'echo "{{\\"task\\": \\"energy_analysis\\", \\"start\\": $_wf_t0, \\"end\\": $_wf_t1}}" >> {_mf}; '
            f'exit $_wf_rc')


def main(dataset_path, output_path, config_path):
    print("Starting demo")

    proteins = [
        (f.split('.')[0], join(dataset_path, f))
        for f in listdir(dataset_path)
        if isfile(join(dataset_path, f))
    ]

    result_futures = []
    for name, pdb in proteins:
        structure = join(output_path, name + '.gro')
        topology = join(output_path, name + '.top')
        f1 = generate_topology(
            protein=pdb, structure=structure, topology=topology,
            inputs=[File(pdb)], outputs=[File(structure), File(topology)],
            stdout=(join(output_path, name + '_topo.stdout'), 'w'),
            stderr=(join(output_path, name + '_topo.stderr'), 'w'),
        )

        structure_newbox = join(output_path, name + '_newbox.gro')
        f2 = define_box(
            structure=structure, structure_newbox=structure_newbox,
            inputs=[f1], outputs=[File(structure_newbox)]
        )

        protein_solv = join(output_path, name + '_solv.gro')
        f3 = add_solvate(
            structure_newbox=structure_newbox, protein_solv=protein_solv, topology=topology,
            inputs=[f2], outputs=[File(protein_solv)]
        )

        ions_conf = join(config_path, 'ions.mdp')
        ions = join(output_path, name + '_ions.tpr')
        f4 = assemble_tpr(
            conf=ions_conf, protein_solv=protein_solv, topology=topology, output=ions,
            inputs=[f3], outputs=[File(ions)]
        )

        protein_solv_ions = join(output_path, name + '_solv_ions.gro')
        group = join(config_path, 'genion.group')
        f5 = replace_solvent_with_ions(
            ions=ions, output=protein_solv_ions, topology=topology, group=group,
            inputs=[f4], outputs=[File(protein_solv_ions)]
        )

        minim_conf = join(config_path, 'minim.mdp')
        em = join(output_path, name + '_em.tpr')
        f6 = assemble_tpr(
            conf=minim_conf, protein_solv=protein_solv_ions, topology=topology, output=em,
            inputs=[f5], outputs=[File(em)]
        )

        em_energy = join(output_path, name + '_em_energy.edr')
        f7 = energy_minimization(
            em=em, em_energy=em_energy,
            inputs=[f6], outputs=[File(em_energy)]
        )

        energy_result = join(output_path, name + '_potential.xvg')
        energy_selection = join(config_path, 'energy.selection')
        f8 = energy_analysis(
            em=em_energy, output=energy_result, selection=energy_selection,
            inputs=[f7], outputs=[File(energy_result)]
        )

        result_futures.append(f8)

    for f in result_futures:
        f.result()

    print("Done")


if __name__ == '__main__':
    config_path = sys.argv[1]
    dataset_path = sys.argv[2]
    output_path = sys.argv[3]
    main(dataset_path, output_path, config_path)
