#!/usr/bin/env python3
import subprocess
from sys import stdout
from unittest import TextTestResult

ITERATION_LIMIT = 150
CONVERGENCE_THRESHOLD = 0.000001
SEED = 8675309
GRADER_THRESHOLD = 0.0000100000001 # floating point errors
EXEC = "./bin/kmeans"
RUNS = 5

METHODS = ["--use_cuda_basic", "--use_cuda_shared"]

INPUT_ANSWERS_FILES = ["input/random-n2048-d16-c16-answer.txt", "input/random-n16384-d24-c16-answer.txt", "input/random-n65536-d32-c16-answer.txt"]
INPUT_FILES = ["input/random-n2048-d16-c16.txt", "input/random-n16384-d24-c16.txt", "input/random-n65536-d32-c16.txt"]
DIMS = [16, 24, 32]


def main():
    # This test file is called by make, so it does not need to make...
    for method in METHODS:
        for input,dims,answer in zip(INPUT_FILES, DIMS,  INPUT_ANSWERS_FILES):
            timing = 0
            for run in range(0, RUNS):
                command_to_run = f"{EXEC} -k 16 -d {dims} -i {input} -m {ITERATION_LIMIT} -t {CONVERGENCE_THRESHOLD} -s {SEED} {method}"
                p = subprocess.getoutput([command_to_run])
                match = True
                cluster = []
                with open(f"{answer}", "r") as f:
                    for point in f.readlines():
                        point = point.replace('\n', '')
                        dim = []
                        for id, dimension in enumerate(point.split(' ')):
                            if id == 0: continue
                            dim.append(float(dimension))
                        cluster.append(dim)

                for p_id, point in enumerate(p.splitlines()):
                    if p_id == 0:
                        timing += float(point.split(',')[1])
                        continue # first line is timing information
                    for d_id, dimension in enumerate(point.split(" ")): # First number is index, we can skip, subsequent DIMS numbers
                        if d_id == 0 or d_id==dims+1: continue # Random space at end so we skip that as well
                        if float(dimension) < (cluster[p_id-1][d_id-1]  - GRADER_THRESHOLD) or float(dimension) > (cluster[p_id-1][d_id-1]  + GRADER_THRESHOLD):
                            print(f"Output: {float(dimension)}, Answer: {cluster[p_id-1][d_id-1] - GRADER_THRESHOLD} - {cluster[p_id-1][d_id-1] + GRADER_THRESHOLD} point_id {p_id} dimension_id {d_id}")
                            match = False
            if not match:
                print(f"{input} does not match for {method}")
                print(f"{command_to_run}")
            else:
                print(f"{command_to_run} : {timing/RUNS} ms, averaged over {RUNS} runs")


if __name__ == "__main__":
    main()