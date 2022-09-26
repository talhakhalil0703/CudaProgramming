#include "kmeans_cpu.h"
#include "random.h"
#include "io.h"
#include <cmath>
#include <limits>

void kmeans_cpu(double **dataset, int clusters, options_t &args) {

// Random assign Centroids
  double **centroids = (double **)malloc(args.num_cluster * sizeof(double *));
  int index = 0;
  for (int i = 0; i < args.num_cluster; i++){
    index = kmeans_rand() % args.number_of_values;
    centroids[i] = dataset[index];
  }
  // print_points(centroids, args.num_cluster ,args.dims);
  int iterations = 0;
  double ** old_centroids = NULL;
  bool done = false;
  int * labels;

  while(!done){
    //copy
    old_centroids = copy_double(centroids, args);

    iterations++;

    //labels is a mapping from each point in the dataset to the enarest euclidian distance centroid
    labels = find_nearest_centroids(dataset, centroids, args);

    // Print Labels
    // for (int i =0 ; i< args.number_of_values; i++){
    //   std::cout << i << ": " << labels[i] << std::endl;
    // }

    //the new centroids are the average of all points that map to each centroid
    centroids = average_labeled_centroids(dataset, labels, clusters, args);

    done = iterations > args.max_num_iter || converged(centroids, old_centroids, args);
    // free old_centroids
    // TODO: Write a freeing function
    for(int i =0; i < args.num_cluster; i++){
      free(old_centroids[i]);
    }
    free(old_centroids);

    // free labels, only if not done
    free (labels);
    printf("Iterations : %d\n", iterations);
    // print_points(centroids, args.num_cluster, args.dims);
  }

  print_points(centroids, args.num_cluster, args.dims);
}

int * find_nearest_centroids(double ** dataset, double ** centroids, options_t &args){
  // For each point we calculate the distance from that point to all centroids, and then store the closest point as the index of the centroid
  // The length of labels here would be the number of points we have...
  int * labels = (int *) calloc (args.number_of_values, sizeof(int));
  double closest_centroid_distance;

  for (int i =0; i < args.number_of_values; i++){
    closest_centroid_distance = std::numeric_limits<double>::max();
    for (int j = 0; j < args.num_cluster; j++){
      double distance = eucledian_distance(dataset[i], centroids[j], args.dims);
      if (distance < closest_centroid_distance){
        closest_centroid_distance = distance;
        labels[i] = j;
      }
    }
  }

  return labels;
}

double ** average_labeled_centroids(double ** dataset, int * labels, int clusters, options_t &args){
  // For a new center given points. Here we need to know how to calculate a centroid.
  double ** centroids = (double **) calloc(args.num_cluster, sizeof(double *));
  int * points_in_centroid = (int *) calloc(args.num_cluster, sizeof(int));

  //For each centroid set the starting point to be zero for each dimensional value
  for (int i =0; i < args.num_cluster; i++){
    double * centroid = (double *) calloc (args.dims, sizeof(double));
    centroids[i] = centroid;
  }

  // For a given centroid define a keep track of all the values associated with that dimension and add them
  // Sum the corresponding points, we also need to know how many points are in each cluster..
  for (int i = 0; i < args.number_of_values; i++){
    int index_of_centroid = labels[i];
    points_in_centroid[index_of_centroid]++;
    for (int j =0 ; j < args.dims; j++){
      centroids[index_of_centroid][j] += dataset[i][j];
    }
  }

  // At the end divide by the total number of elements for each dimension
  for(int i = 0; i < args.num_cluster; i++){
    for (int j = 0; j < args.dims; j++){
      centroids[i][j] /= points_in_centroid[i];
    }
  }

  //print the points in each centroid
  printf("Points in Centroid\n");
  for (int i =0; i< args.num_cluster; i++){
    printf("%d:%d ", i, points_in_centroid[i] );
  }
  printf("\n");

  free(points_in_centroid);
  return centroids;
}

bool converged(double ** new_centroids, double** old_centroids, options_t &args) {
  double distance = std::numeric_limits<double>::max();
  for (int i =0; i < args.num_cluster; i++){
    distance = eucledian_distance(new_centroids[i], old_centroids[i], args.dims);
    if (distance > args.threshold) return false;
  }
  // Check if each of the centroid has moved less than the threshold provided.
  return true;
}

double eucledian_distance(double * first, double * second, int dimensions){
  double sum = 0;
  for (int i = 0; i < dimensions; i++){
    sum += pow(first[i]-second[i], 2.0);
  }
  return sqrt(sum);
}

double ** copy_double(double ** original, options_t args)
{
  double ** copy = (double **) malloc(args.num_cluster * sizeof(double*));
  for (int i = 0; i < args.num_cluster; i++){
    double * inner = (double *) malloc (args.dims * sizeof(double));
    copy[i] = inner;
  }

  for (int i =0; i < args.num_cluster; i++){
    for (int j = 0; j< args.dims; j++){
      copy[i][j] = original[i][j];
    }
  }
  return copy;
}