#include "argparse.h"
#include <cuda_runtime.h>

void kmeans_cuda_basic(float *dataset, float * centroids, options_t &args);

bool cuda_converged(float * new_centroids, float* old_centroids, options_t &args);
__global__ void d_cuda_convergence_helper(float * new_c, float * old_c, float * temp, int dimensions, int num_cluster);
__global__ void d_cuda_convergence_helper_threshold(float * temp, int * converged, int num_cluster, float threshold);

void cuda_find_nearest_centroids(float * dataset,int * labels,  float * centroids, options_t &args);
__global__ void d_cuda_find_nearest_centroids(float * dataset, float * centroids, int * labels, int dims, int num_centroids, float max, int num_v);

void cuda_average_labeled_centroids(float * d_centroids, float * d_dataset, int * h_labels, options_t &args);
__global__ void d_cuda_average_labeled_centroids(float * dataset, int * labels, int * points_in_centroids, float * centroids, int number_of_values, int dims, int num_cluster);
__global__ void d_cuda_average_labeled_centroids_divide(float * centroids, int * points_in_centroids, int dims, int num_cluster);




