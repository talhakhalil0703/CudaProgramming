#include "argparse.h"
#include <cuda_runtime.h>

void kmeans_cuda_basic(double *dataset, double * centroids, options_t &args);
double * cuda_copy(double * original, options_t args);

bool cuda_converged(double * new_centroids, double* old_centroids, options_t &args);
__global__ void d_cuda_convergence_helper(double * new_c, double * old_c, double * temp, int dimensions, int num_cluster);
__global__ void d_cuda_convergence_helper_threshold(double * temp, int * converged, int num_cluster, double threshold);
void cuda_find_nearest_centroids(double * dataset,int * labels,  double * centroids, options_t &args);
__global__ void d_cuda_find_nearest_centroids(double * dataset, double * centroids, int * labels, int dims, int num_centroids, double max, int num_v);
void cuda_average_labeled_centroids(double * d_centroids, double * d_dataset, int * h_labels, options_t &args);
__global__ void d_cuda_average_labeled_centroids(double * dataset, int * labels, int * points_in_centroids, double * centroids, int number_of_values, int dims, int num_cluster);
__global__ void d_cuda_average_labeled_centroids_divide(double * centroids, int * points_in_centroids, int dims, int num_cluster);




