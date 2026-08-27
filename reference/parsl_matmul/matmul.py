import argparse
import json
import os
import threading
import time

import parsl
from parsl import python_app
from parsl.config import Config
from parsl.executors import ThreadPoolExecutor

config = Config(executors=[ThreadPoolExecutor(max_threads=4)])
parsl.load(config)

# --- instrumentation added for makespan / cpu / parallelism-efficiency analysis ---
_METRICS_FILE = os.environ.get("WF_METRICS_FILE", "metrics.jsonl")
_METRICS_LOCK = threading.Lock()


def _record(name, t0, t1):
    line = json.dumps({"task": name, "start": t0, "end": t1, "duration": t1 - t0})
    with _METRICS_LOCK:
        with open(_METRICS_FILE, "a") as f:
            f.write(line + "\n")


def _timed(name):
    def deco(fn):
        def wrapper(*args, **kwargs):
            t0 = time.time()
            try:
                return fn(*args, **kwargs)
            finally:
                _record(name, t0, time.time())
        wrapper.__name__ = fn.__name__
        return wrapper
    return deco
# --- end instrumentation ---


@python_app
@_timed("generate_block")
def generate_block(size, num_blocks, seed=0, set_to_zero=False):
    import numpy as np
    np.random.seed(seed)
    if not set_to_zero:
        b = np.random.random((size, size))
        b /= np.sum(b) * float(num_blocks)
    else:
        b = np.zeros((size, size))
    return b


@python_app
@_timed("fused_multiply_add")
def fused_multiply_add(A, B, C):
    import numpy as np
    return C + np.dot(A, B)


def dot(A, B, C):
    n, m = len(A), len(B[0])
    for i in range(n):
        for j in range(m):
            for k in range(n):
                C[i][j] = fused_multiply_add(A[i][k], B[k][j], C[i][j])


def main(num_blocks, elems_per_block, seed):
    start_time = time.time()

    A, B, C = [], [], []
    for i in range(num_blocks):
        A.append([])
        B.append([])
        C.append([])
        bid = 0
        for j in range(num_blocks):
            A[-1].append(generate_block(elems_per_block, num_blocks, seed=seed + bid))
            bid += 1
            B[-1].append(generate_block(elems_per_block, num_blocks, seed=seed + bid))
            bid += 1
            C[-1].append(generate_block(elems_per_block, num_blocks, set_to_zero=True))

    for i in range(num_blocks):
        for j in range(num_blocks):
            A[i][j] = A[i][j].result()
            B[i][j] = B[i][j].result()
            C[i][j] = C[i][j].result()

    initialization_time = time.time()

    dot(A, B, C)

    for i in range(num_blocks):
        for j in range(num_blocks):
            C[i][j] = C[i][j].result()

    multiplication_time = time.time()

    print("-----------------------------------------")
    print("-------------- RESULTS ------------------")
    print("-----------------------------------------")
    print("Initialization time: %f" % (initialization_time - start_time))
    print("Multiplication time: %f" % (multiplication_time - initialization_time))
    print("Total time: %f" % (multiplication_time - start_time))
    print("-----------------------------------------")


def parse_args():
    parser = argparse.ArgumentParser(description='Parsl blocked matmul implementation')
    parser.add_argument('-b', '--num_blocks', type=int, default=1)
    parser.add_argument('-e', '--elems_per_block', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


if __name__ == '__main__':
    opts = parse_args()
    main(**vars(opts))
