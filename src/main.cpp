#include <iostream>
#include <chrono>
#include <cstring>
#include "argparse.h"
#include "random.h"

using namespace std;

int main(int argc, char **argv)
{
    // Parse args
    struct options_t opts;
    get_opts(argc, argv, &opts);

    // Start timer
    auto start = std::chrono::high_resolution_clock::now();
    print_opts(&opts);
    // Set the seed for random.
    kmeans_set_rand_seed(opts.seed);

    //End timer and print out elapsed
    auto end = std::chrono::high_resolution_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "time: " << diff.count() << " us" <<std::endl;

}
