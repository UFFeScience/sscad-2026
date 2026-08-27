import json
import time
import numpy as np

from pycompss.api.task import task
from pycompss.api.parameter import INOUT
from pycompss.api.api import compss_barrier
from pycompss.api.api import compss_wait_on


# --- instrumentation added for makespan / cpu / parallelism-efficiency analysis ---
def _record(name, t0, t1):
    # Hardcoded absolute path: COMPSs launches workers over SSH to localhost, which does
    # not propagate the parent shell's environment variables to the worker process.
    mf = "/workspace/matmul/metrics.jsonl"
    line = json.dumps({"task": name, "start": t0, "end": t1, "duration": t1 - t0})
    with open(mf, "a") as f:
        f.write(line + "\n")
# --- end instrumentation ---


@task(returns=1)
def generate_block(size, num_blocks, seed=0, set_to_zero=False):
    _t0 = time.time()
    np.random.seed(seed)
    if not set_to_zero:
        b = np.random.random((size, size))
        b /= np.sum(b) * float(num_blocks)
    else:
        b = np.zeros((size, size))
    _record("generate_block", _t0, time.time())
    return b


@task(C=INOUT)
def fused_multiply_add(A, B, C):
    _t0 = time.time()
    C += np.dot(A, B)
    _record("fused_multiply_add", _t0, time.time())


def dot(A, B, C):
    n, m = len(A), len(B[0])
    for i in range(n):
        for j in range(m):
            for k in range(n):
                fused_multiply_add(A[i][k], B[k][j], C[i][j])


def main(num_blocks, elems_per_block, seed):
    start_time = time.time()

    A, B, C = [], [], []
    matrix_name = ["A", "B"]
    for i in range(num_blocks):
        for l in [A, B, C]:
            l.append([])
        bid = 0
        for j in range(num_blocks):
            for ix, l in enumerate([A, B]):
                l[-1].append(generate_block(elems_per_block, num_blocks, seed + bid, False))
                bid += 1
            C[-1].append(generate_block(elems_per_block, num_blocks, 0, True))
    compss_barrier()
    initialization_time = time.time()

    dot(A, B, C)

    compss_barrier()
    multiplication_time = time.time()

    print("-----------------------------------------")
    print("-------------- RESULTS ------------------")
    print("-----------------------------------------")
    print("Initialization time: %f" % (initialization_time - start_time))
    print("Multiplication time: %f" % (multiplication_time - initialization_time))
    print("Total time: %f" % (multiplication_time - start_time))
    print("-----------------------------------------")


def parse_args():
    import argparse
    description = 'COMPSs blocked matmul implementation'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-b', '--num_blocks', type=int, default=1)
    parser.add_argument('-e', '--elems_per_block', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    opts = parse_args()
    main(**vars(opts))
