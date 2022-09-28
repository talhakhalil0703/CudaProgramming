#include "argparse.h"
#include <cuda_runtime.h>

void kmeans_cuda_basic(double **dataset, int clusters, options_t &args);
double ** cuda_copy_double(double ** original, options_t args);

double cuda_eucledian_distance(double * first, double * second, int dimensions);
__global__ void d_eucledian_distance_helper(double * first, double * second, double * pow, double * ret, int dim);
bool cuda_converged(double ** new_centroids, double** old_centroids, options_t &args);
__global__ void d_cuda_convergence_helper(double * new_c, double * old_c, double * temp, bool * convergence, double threshold, int dimensions);
int * cuda_find_nearest_centroids(double ** dataset, double ** centroids, options_t &args);
__global__ void d_cuda_find_nearest_centroids(double * dataset, double * centroids, double * temp, int * labels, int dims, double max);
double ** cuda_average_labeled_centroids(double ** dataset, int * labels, options_t &args);
__global__ void d_cuda_average_labeled_centroids(double * d_dataset, int * d_labels, double * centroids, int number_of_values);

