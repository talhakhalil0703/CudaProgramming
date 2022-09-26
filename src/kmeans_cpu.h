#include "argparse.h"

void kmeans_cpu(double **dataset, int clusters, options_t &args);
int * find_nearest_centroids(double ** dataset, double ** centroids);
double ** average_labeled_centroids(double ** dataset, int * labels, int clusters);
bool converged(double ** new_centroids, double** old_centroids);
