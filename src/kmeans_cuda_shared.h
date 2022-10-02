#include "argparse.h"
#include <cuda_runtime.h>

void kmeans_cuda_shared(float *dataset, float * centroids, options_t &args);
float * cuda_shared_copy(float * original, options_t args);

bool cuda_shared_converged(float * new_centroids, float* old_centroids, options_t &args, float * duration);
__global__ void d_cuda_shared_convergence_helper(float * new_c, float * old_c, bool * convergence, float threshold, int dimensions);
int * cuda_shared_find_nearest_centroids(float * dataset, float * centroids, options_t &args, float * duration);
__global__ void d_cuda_shared_find_nearest_centroids(float * dataset, float * centroids, int * labels, int dims, float max);
float * cuda_shared_average_labeled_centroids(float * dataset, int * labels, options_t &args, float * duration);
__global__ void d_cuda_shared_average_labeled_centroids(float * d_dataset, int * d_labels, float * centroids, int number_of_values);

