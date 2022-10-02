#include "argparse.h"

void kmeans_thrust(float *dataset, float * centroids, options_t &args);

bool thrust_converged(float * new_centroids, float* old_centroids, options_t &args, float * duration);
__global__ void thrust_convergence_helper(float * new_c, float * old_c, bool * convergence, float threshold, int dimensions);
int * thrust_find_nearest_centroids(float * dataset, float * centroids, options_t &args, float * duration);
__global__ void thrust_find_nearest_centroids_helper(float * dataset, float * centroids, int * labels, int dims, float max);
float * thrust_average_labeled_centroids(float * dataset, int * labels, options_t &args, float * duration);
__global__ void thrust_average_labeled_centroids_helper(float * d_dataset, int * d_labels, float * centroids, int number_of_values);

float * copy_data(float * original, options_t args);
