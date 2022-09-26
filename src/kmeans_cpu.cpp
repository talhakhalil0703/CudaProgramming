#include "random.h"
#include "io.h"

void kmeans_cpu(double **dataset, int clusters, int number_of_values, options_t &args) {
  double **centeroids;
  centeroids = (double **)malloc(args.num_cluster * sizeof(double *));
  int index = 0;
  for (int i = 0; i < args.num_cluster; i++){
    index = kmeans_rand() % number_of_values;
    centeroids[i] = dataset[index];
  }
    //print_points(centeroids, args.num_cluster ,args.dims);
}