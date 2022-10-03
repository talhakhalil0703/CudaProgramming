#include <iostream>
#include <chrono>
#include <cstring>
#include "argparse.h"
#include "random.h"
#include "io.h"
#include "kmeans_cuda_basic.h"
#include "kmeans_cuda_shared.h"
#include "kmeans_thrust.h"
#include "kmeans_cpu.h"

using namespace std;

int main(int argc, char **argv)
{
    // Parse args
    struct options_t opts;
    get_opts(argc, argv, &opts);

    // Read the input file
    float * vals;
    read_file(&opts, &vals);

    // Random assign Centroids
    // Set the seed for random.
    kmeans_set_rand_seed(opts.seed);

    float * centroids = (float *) malloc(opts.num_cluster * opts.dims * sizeof(float));
    for (int i = 0, index = 0; i < opts.num_cluster; i++){
        index = kmeans_rand() % opts.number_of_values;
        for (int j =0 ; j < opts.dims; j++){
            centroids[i*opts.dims + j] = vals[index*opts.dims + j];
        }
    }

    if (opts.use_cpu){
        kmeans_cpu(vals, centroids, opts);
    } else if (opts.use_cuda_shared){
        kmeans_cuda_shared(vals, centroids, opts);
    } else if (opts.use_cuda_basic){
        kmeans_cuda_basic(vals, centroids, opts);
    } else if (opts.use_thrust){
        kmeans_thrust(vals, centroids, opts);
    }

    if (!opts.c){
        printf("clusters:");
        for (int p=0; p < opts.number_of_values; p++) {
            printf(" %d", opts.labels[p]);
        }
    } else {
        for (int clusterId = 0; clusterId < opts.num_cluster; clusterId ++){
            printf("%d ", clusterId);
            for (int d = 0; d < opts.dims; d++){
                printf("%lf ", opts.centroids[clusterId*opts.dims  + d ]);
            }
            printf("\n");
        }
    }
    free(vals);
}
