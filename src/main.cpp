#include <iostream>
#include <chrono>
#include <cstring>
#include "argparse.h"
#include "random.h"
#include "io.h"
#include "kmeans_cuda_basic.h"
#include "kmeans_cpu.h"

using namespace std;

int main(int argc, char **argv)
{
    // Parse args
    struct options_t opts;
    get_opts(argc, argv, &opts);

    // Start timer
    auto start = std::chrono::high_resolution_clock::now();
    // Set the seed for random.
    kmeans_set_rand_seed(opts.seed);
    double ** vals;
    read_file(&opts, &vals, &vals);
    if (opts.use_cpu){
        std::cout << "Using CPU: " << std::endl;
        kmeans_cpu(vals, opts.num_cluster, opts);
    } else if (opts.use_cuda_basic){
        std::cout << "Using CUDA Basic: " << std::endl;
        kmeans_cuda_basic(vals, opts.num_cluster, opts);
    }

    if (opts.c){
        printf("clusters:");
        for (int p=0; p < opts.number_of_values; p++) {
            printf(" %d", opts.labels[p]);
        }
    } else {
        if (!opts.use_cpu) {

        // TODO : Wont work for CPU because CPU is still 2D
        for (int clusterId = 0; clusterId < opts.num_cluster; clusterId ++){
            printf("%d ", clusterId);
            for (int d = 0; d < opts.dims; d++){
                printf("%0.5lf ", opts.centroids[clusterId*opts.dims  + d ]);
            }
            printf("\n");
        }
        }
    }
    //End timer and print out elapsed
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::microseconds diff = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "time: " << diff.count() << " us" <<std::endl;
    free_input_points(vals, opts.number_of_values);
}
