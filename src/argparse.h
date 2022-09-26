#ifndef _ARGPARSE_H
#define _ARGPARSE_H

#include <getopt.h>
#include <stdlib.h>
#include <iostream>

struct options_t {
    int num_cluster;
    int dims;
    char *inputfilename; // This naming style was a requirement for the grader
    int max_num_iter;
    double threshold;
    int seed;
    bool c;
    bool use_cpu;
    bool use_cuda_shared;
    bool use_cuda_basic;
    bool use_thrust;
    int number_of_values;
};

void get_opts(int argc, char **argv, struct options_t *opts);
void print_opts(struct options_t *opts);
#endif
