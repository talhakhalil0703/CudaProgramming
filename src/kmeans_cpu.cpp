#include "random.h"
#include "io.h"

void kmeans_cpu(double **dataset, int clusters, options_t &args) {

// Random assign Centroids
  double **centroids;
  centroids = (double **)malloc(args.num_cluster * sizeof(double *));
  int index = 0;
  for (int i = 0; i < args.num_cluster; i++){
    index = kmeans_rand() % args.number_of_values;
    centroids[i] = dataset[index];
  }
  print_points(centroids, args.num_cluster ,args.dims);
  // int iterations = 0;
  // double ** old_centroids = NULL;
  // bool done = false;
  // int * labels;

  // while(!done){
  //   old_centroids = centroids;
  //   iterations++;

  //   //labels is a mapping from each point in the dataset to the enarest euclidian distance centroid
  //   labels = find_nearest_centroids(dataset, centroids);

  //   //the new centroids are the average of all points that map to each centroid
  //   centroids = average_labeled_centroids(dataset, labels, clusters);
  //   done = iterations > args.max_num_iter || converged(centroids, old_centroids);
  // }

}

int * find_nearest_centroids(double ** dataset, double ** centroids){
  // For each point we calculate the distance from that point to all centroids, and then store the closest point as the index of the centroid
  // The length of labels here would be the number of points we have...
  // Need to know number of points as well as
return nullptr;
}

double ** average_labeled_centroids(double ** dataset, int * labels, int clusters){
  // For a new center given points. Here we need to know how to calculate a centroid.
return nullptr;
}

bool converged(double ** new_centroids, double** old_centroids) {
return false;
}