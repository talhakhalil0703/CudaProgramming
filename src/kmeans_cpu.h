#include "argparse.h"

void kmeans_cpu(double **dataset, int clusters, options_t &args);
int * find_nearest_centroids(double ** dataset, double ** centroids, options_t &args);
double ** average_labeled_centroids(double ** dataset, int * labels, int clusters, options_t &args);
bool converged(double ** new_centroids, double** old_centroids, options_t &args);
double eucledian_distance(double * first, double * second, int dimensions);
double ** copy_double(double ** original, options_t args);
