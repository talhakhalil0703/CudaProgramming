#include "argparse.h"

void kmeans_cpu(float * dataset, float * centroids, options_t &args);
int * find_nearest_centroids(float * dataset, float * centroids, options_t &args);
float * average_labeled_centroids(float * dataset, int * labels, options_t &args);
bool converged(float * new_centroids, float* old_centroids, options_t &args);
float eucledian_distance(float * first, float * first_end, float * second,float * second_end, int dimensions);
float * seq_copy(float * original, options_t args);
