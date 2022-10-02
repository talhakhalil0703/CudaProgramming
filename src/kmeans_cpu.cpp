#include "kmeans_cpu.h"
#include "random.h"
#include "io.h"
#include <cmath>
#include <limits>
#include <chrono>

void kmeans_cpu(float *dataset, float * centroids, options_t &args) {

  int iterations = 0;
  float * old_centroids = NULL;
  bool done = false;
  int * labels;
  float duration_total = 0;

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
    centroids = average_labeled_centroids(dataset, labels, args);

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

int * find_nearest_centroids(float * dataset, float * centroids, options_t &args){
  // For each point we calculate the distance from that point to all centroids, and then store the closest point as the index of the centroid
  // The length of labels here would be the number of points we have...
  int * labels = (int *) calloc (args.number_of_values, sizeof(int));
  float closest_centroid_distance;

  for (int i =0; i < args.number_of_values; i++){
    closest_centroid_distance = std::numeric_limits<float>::max();
    float distance = 0;
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

float * average_labeled_centroids(float * dataset, int * labels, options_t &args){
  // For a new center given points. Here we need to know how to calculate a centroid.
  float * centroids = (float *) calloc(args.num_cluster * args.dims, sizeof(float));
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

bool converged(float * new_centroids, float* old_centroids, options_t &args) {
  float distance = std::numeric_limits<float>::max();
  for (int i =0; i < args.num_cluster; i++){
    distance = eucledian_distance(&new_centroids[i*args.dims],&new_centroids[(i+1)*args.dims], &old_centroids[i*args.dims], &old_centroids[(i+1)*args.dims], args.dims);
    if (distance > args.threshold) return false;
  }
  // Check if each of the centroid has moved less than the threshold provided.
  return true;
}

float eucledian_distance(float * first_start, float * first_end, float * second_start,float * second_end, int dimensions){
  float sum = 0;
  float * first_tracker = first_start;
  float * second_tracker = second_start;
  for (;first_tracker != first_end; first_tracker++, second_tracker++){
    sum += pow(*first_tracker-*second_tracker, 2.0);
  }
  // sum += pow(first_end-second_end, 2.0);

  return sqrt(sum);
}

float * seq_copy(float * original, options_t args)
{
  float * copy = (float *) malloc(args.num_cluster * args.dims * sizeof(float));

  for (int i =0; i < args.num_cluster * args.dims; i++){
    copy[i] = original[i];
  }

  return copy;
}
