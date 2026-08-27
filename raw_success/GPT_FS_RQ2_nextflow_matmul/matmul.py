import argparse
import numpy as np

parser=argparse.ArgumentParser()

parser.add_argument("--A")
parser.add_argument("--B")
parser.add_argument("--output")

args=parser.parse_args()

A = np.loadtxt(args.A, delimiter=",")
B = np.loadtxt(args.B, delimiter=",")

C = A @ B

np.savetxt(args.output, C, delimiter=",", fmt="%.6f")