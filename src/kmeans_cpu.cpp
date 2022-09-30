#include "kmeans_cpu.h"
#include "random.h"
#include "io.h"
#include <cmath>
#include <limits>
#include <chrono>

void kmeans_cpu(double **d_dataset, int clusters, options_t &args) {

// Random assign Centroids
  double **d_centroids = (double **)malloc(args.num_cluster * sizeof(double *));
  int index = 0;
  for (int i = 0; i < args.num_cluster; i++){
    index = kmeans_rand() % args.number_of_values;
    d_centroids[i] = d_dataset[index];
  }
  //Conversion to 1D arrays

  double * dataset = (double *) malloc (args.number_of_values * args.dims * sizeof(double));
  double * centroids = (double *) malloc (args.num_cluster * args.dims * sizeof(double));

  index = 0;
  for (int i = 0; i< args.number_of_values; i++){
    for (int j =0; j < args.dims; j++){
      dataset[index++] = d_dataset[i][j];
    }
  }

  index = 0;
  for (int i = 0; i< args.num_cluster; i++){
    for (int j =0; j < args.dims; j++){
      centroids[index++] = d_centroids[i][j];
    }
  }

  // print_points(centroids, args.num_cluster ,args.dims);
  int iterations = 0;
  double * old_centroids = NULL;
  bool done = false;
  int * labels;
  double duration_total = 0;

  while(!done){
    //copy
    auto start = std::chrono::high_resolution_clock::now();

    old_centroids = seq_copy(centroids, args);

    iterations++;

    //labels is a mapping from each point in the dataset to the enarest euclidian distance centroid
    labels = find_nearest_centroids(dataset, centroids, args);

    //Print Labels
    // for (int i =0 ; i< args.number_of_values; i++){
    //   std::cout << i << ": " << labels[i] << std::endl;
    // }

    //the new centroids are the average of all points that map to each centroid
    centroids = average_labeled_centroids(dataset, labels, clusters, args);

    done = iterations > args.max_num_iter || converged(centroids, old_centroids, args);

    // free old_centroids
    free(old_centroids);

    // free labels, only if not done
    auto end = std::chrono::high_resolution_clock::now();
    int duration = std::chrono::duration_cast<std::chrono::milliseconds>( end - start ).count();
    duration_total += duration;
    if (!done) free (labels);
  }
  args.labels = labels;
  args.centroids = centroids;
  printf("%d,%lf\n", iterations, duration_total/iterations);
}

int * find_nearest_centroids(double * dataset, double * centroids, options_t &args){
  // For each point we calculate the distance from that point to all centroids, and then store the closest point as the index of the centroid
  // The length of labels here would be the number of points we have...
  int * labels = (int *) calloc (args.number_of_values, sizeof(int));
  double closest_centroid_distance;

  for (int i =0; i < args.number_of_values; i++){
    closest_centroid_distance = std::numeric_limits<double>::max();
    double distance = 0;
    for (int j =0 ; j < args.num_cluster; j++){
      distance = eucledian_distance(&dataset[i*args.dims], &dataset[(i+1)*args.dims], &centroids[(j)*args.dims], &centroids[(j+1)*args.dims], args.dims);
      if (distance <  closest_centroid_distance){
        closest_centroid_distance = distance;
        labels[i] = j;
      }
    }
  }


  return labels;
}

double * average_labeled_centroids(double * dataset, int * labels, int clusters, options_t &args){
  // For a new center given points. Here we need to know how to calculate a centroid.
  double * centroids = (double *) calloc(args.num_cluster * args.dims, sizeof(double));
  int * points_in_centroid = (int *) calloc(args.num_cluster, sizeof(int));


  // For a given centroid define a keep track of all the values associated with that dimension and add them
  // Sum the corresponding points, we also need to know how many points are in each cluster..
  for (int i = 0; i < args.number_of_values; i++){
    int index_of_centroid = labels[i];
    points_in_centroid[index_of_centroid]++;
    for (int j =0 ; j < args.dims; j++){
      centroids[index_of_centroid*args.dims + j] += dataset[i*args.dims + j];
    }
  }

  // At the end divide by the total number of elements for each dimension
  for(int i = 0; i < args.num_cluster; i++){
    for (int j = 0; j < args.dims; j++){
      centroids[i*args.dims + j] /= points_in_centroid[i];
    }
  }

  free(points_in_centroid);
  return centroids;
}

bool converged(double * new_centroids, double* old_centroids, options_t &args) {
  double distance = std::numeric_limits<double>::max();
  for (int i =0; i < args.num_cluster; i++){
    distance = eucledian_distance(&new_centroids[i*args.dims],&new_centroids[(i+1)*args.dims], &old_centroids[i*args.dims], &old_centroids[(i+1)*args.dims], args.dims);
    if (distance > args.threshold) return false;
  }
  // Check if each of the centroid has moved less than the threshold provided.
  return true;
}

double eucledian_distance(double * first_start, double * first_end, double * second_start,double * second_end, int dimensions){
  double sum = 0;
  double * first_tracker = first_start;
  double * second_tracker = second_start;
  for (;first_tracker != first_end; first_tracker++, second_tracker++){
    sum += pow(*first_tracker-*second_tracker, 2.0);
  }
  // sum += pow(first_end-second_end, 2.0);

  return sqrt(sum);
}

double * seq_copy(double * original, options_t args)
{
  double * copy = (double *) malloc(args.num_cluster * args.dims * sizeof(double));

  for (int i =0; i < args.num_cluster * args.dims; i++){
    copy[i] = original[i];
  }

  return copy;
}
