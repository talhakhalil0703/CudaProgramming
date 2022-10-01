#include "argparse.h"

void kmeans_thrust(double *dataset, double * centroids, options_t &args);

bool thrust_converged(double * new_centroids, double* old_centroids, options_t &args, double * duration);
__global__ void thrust_convergence_helper(double * new_c, double * old_c, bool * convergence, double threshold, int dimensions);
int * thrust_find_nearest_centroids(double * dataset, double * centroids, options_t &args, double * duration);
__global__ void thrust_find_nearest_centroids_helper(double * dataset, double * centroids, int * labels, int dims, double max);
double * thrust_average_labeled_centroids(double * dataset, int * labels, options_t &args, double * duration);
__global__ void thrust_average_labeled_centroids_helper(double * d_dataset, int * d_labels, double * centroids, int number_of_values);

double * copy_data(double * original, options_t args);
