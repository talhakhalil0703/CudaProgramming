#include "kmeans_cuda_basic.h"
#include "random.h"
#include "io.h"
// #include <cmath>
#include <limits>
#include <chrono>
#include <math.h>

#define NUMBER_OF_THREADS 1024

void kmeans_cuda_basic(float *dataset, float * centroids, options_t &args) {
  int iterations = 0;
  bool done = false;
  float duration_total = 0;
  float duration = 0;

  int * d_labels;
  cudaMalloc((void**)&d_labels, args.number_of_values * sizeof(int));
  cudaMemset(d_labels, 0, args.number_of_values * sizeof(int)); // Should start from zero?

  float * d_dataset;
  cudaMalloc((void**)&d_dataset, args.number_of_values * args.dims * sizeof(float));
  cudaMemcpy(d_dataset, dataset, args.number_of_values * args.dims * sizeof(float), cudaMemcpyHostToDevice);

  float * d_centroids;
  cudaMalloc((void**)&d_centroids, args.num_cluster * args.dims * sizeof(float));
  cudaMemcpy(d_centroids, centroids, args.num_cluster * args.dims * sizeof(float), cudaMemcpyHostToDevice);

  float * old_centroids;
  cudaMalloc((void**)&old_centroids, args.num_cluster * args.dims * sizeof(float));

  cudaEvent_t start_t, stop_t;
  cudaEventCreate(&start_t);
  cudaEventCreate(&stop_t);
  cudaEventRecord(start_t);

  while(!done){
    //copy
    duration = 0;

    cudaMemcpy(old_centroids, d_centroids, args.num_cluster * args.dims * sizeof(float), cudaMemcpyDeviceToDevice);

    iterations++;

    //labels is a mapping from each point in the dataset to the enarest euclidian distance centroid
    cuda_find_nearest_centroids(d_dataset, d_labels, d_centroids, args);

    //the new centroids are the average of all points that map to each centroid
    cuda_average_labeled_centroids(d_centroids, d_dataset, d_labels, args);

    done = iterations > args.max_num_iter || cuda_converged(d_centroids, old_centroids, args);

    duration_total += duration;
  }
  cudaEventRecord(stop_t);
  cudaDeviceSynchronize();

  float total_time = 0;
  cudaEventElapsedTime(&total_time, start_t, stop_t);

  printf("%d,%f\n", iterations, total_time/iterations);

  int * labels;
  labels = (int *) malloc(args.number_of_values*sizeof(int));
  cudaMemcpy(labels, d_labels, args.number_of_values*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(centroids,d_centroids, args.num_cluster * args.dims * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(old_centroids);
  cudaFree(d_labels);
  cudaFree(d_centroids);
  cudaFree(d_dataset);

  args.labels = labels;
  args.centroids = centroids;
}

void cuda_find_nearest_centroids(float * d_dataset, int * d_labels, float * d_centroids, options_t &args){
  //Launch the kernel
  cudaEvent_t start_t, stop_t;
  cudaEventCreate(&start_t);
  cudaEventCreate(&stop_t);
  int num_blocks = args.number_of_values/NUMBER_OF_THREADS;
  if (num_blocks == 0) num_blocks = 1;
  cudaEventRecord(start_t);
  d_cuda_find_nearest_centroids<<<dim3(num_blocks), dim3(NUMBER_OF_THREADS)>>>(d_dataset, d_centroids, d_labels, args.dims, args.num_cluster, std::numeric_limits<float>::max(), args.number_of_values);
  //Sync
  cudaEventRecord(stop_t);
  cudaDeviceSynchronize();

  // float total_time = 0;
  // cudaEventElapsedTime(&total_time, start_t, stop_t);
  // std::cout << "find_nearest_centroids: " << total_time << std::endl;

}

__global__ void d_cuda_find_nearest_centroids(float * dataset, float * centroids, int * labels, int dims, int num_cluster, float max, int number_of_values){
  // Each thread is given a point and for each point we want to find the closest centroid.
  int index = threadIdx.x + blockIdx.x * blockDim.x;

  // Ensure that the point is actually a point that exists
  if (index <  number_of_values){
    float shortest_distance = max;
    float current_distance = 0;
    int closest_index = 0;

    for (int i = 0; i < num_cluster; i++){
      current_distance = 0;
      for (int j =0 ; j < dims; j++){
        current_distance += powf(dataset[index*dims + j] - centroids[i*dims + j], 2.0);
      }
      current_distance = sqrtf(current_distance);
      if (current_distance < shortest_distance){
        shortest_distance = current_distance;
        closest_index = i;
      }
    }

    labels[index] = closest_index;
  }
}

void cuda_average_labeled_centroids(float * d_centroids, float * d_dataset, int * d_labels, options_t &args){
  // Allocate Device Memory
  cudaEvent_t start_t, stop_t;
  cudaEventCreate(&start_t);
  cudaEventCreate(&stop_t);

  int * d_points_in_centroids;
  cudaMalloc ((void **)&d_points_in_centroids, args.num_cluster*sizeof(int));

  // Transfer Memory From Host To Device
  cudaMemset(d_centroids, 0, args.num_cluster * args.dims * sizeof(float)); // Should start from zero?
  cudaMemset(d_points_in_centroids, 0, args.num_cluster * sizeof(int)); // Should start from zero?

  // Launch the kernel
  int num_blocks = args.number_of_values/NUMBER_OF_THREADS;
  if (num_blocks == 0) num_blocks = 1;
  cudaEventCreate(&start_t);

  d_cuda_average_labeled_centroids<<<dim3(num_blocks), dim3(NUMBER_OF_THREADS)>>>(d_dataset, d_labels, d_points_in_centroids, d_centroids, args.number_of_values, args.dims, args.num_cluster);

  // Sync
  cudaEventRecord(stop_t);
  cudaDeviceSynchronize();
  // float total_time = 0;
  // cudaEventElapsedTime(&total_time, start_t, stop_t);
  // std::cout << "averaged_labeled_centroids: " << total_time << std::endl;

  num_blocks = args.num_cluster*args.dims/NUMBER_OF_THREADS;
  if (num_blocks == 0) num_blocks = 1;
  cudaEventCreate(&start_t);
  d_cuda_average_labeled_centroids_divide<<<num_blocks,NUMBER_OF_THREADS>>>(d_centroids, d_points_in_centroids, args.dims, args.num_cluster);
  // Sync
  cudaEventRecord(stop_t);

  cudaDeviceSynchronize();
  // total_time = 0;
  // cudaEventElapsedTime(&total_time, start_t, stop_t);
  // std::cout << "averaged_labeled_centroids 3: " << total_time << std::endl;

  // Free Device Memory
  cudaFree(d_points_in_centroids);

}

__global__ void d_cuda_average_labeled_centroids(float * dataset, int * labels, int * points_in_centroids, float * centroids, int number_of_values, int dims, int num_cluster){
  // Unique index for each point
  int index = threadIdx.x + blockIdx.x * blockDim.x;

  //Check to ensure that the index is actually less than the total number of values we have
  if (index <  number_of_values) {
    //Get the centroid that the point belongs to.
    int centroid_index = labels[index];

    //Increment the count of points that centroid contains.
    atomicAdd(&points_in_centroids[centroid_index], 1);

    //Increment that centroids dims with that point
    for (int i = 0 ; i < dims;  i++){
      atomicAdd(&centroids[centroid_index*dims + i], dataset[index*dims+i]);
    }

  }
}

__global__ void d_cuda_average_labeled_centroids_divide(float * centroids, int * points_in_centroids, int dims, int num_cluster){
  int index = threadIdx.x + blockIdx.x * blockDim.x;
   if (index < dims*num_cluster) {
    //For each centroid divide it's dimensions with the amount of points present.
    //We can store this value
    int points_in_centroid = points_in_centroids[index/dims];

    centroids[index] /= points_in_centroid;
   }


}

bool cuda_converged(float * d_new_centroids, float* d_old_centroids, options_t &args) {

  int * h_converged = (int *) malloc(sizeof(int));

  // Allocate Device Memory
  cudaEvent_t start_t, stop_t;
  cudaEventCreate(&start_t);
  cudaEventCreate(&stop_t);

  //Allocate Device Memory
  float * d_intermediate_values;
  int * d_converged;

  cudaMalloc((void**)&d_intermediate_values, args.num_cluster*sizeof(float));
  cudaMalloc((void**)&d_converged, sizeof(int));

  // Transfer Memory from Host to Device
  cudaMemset(d_intermediate_values, 0, args.num_cluster * sizeof(float)); // Should start from zero?
  cudaMemset(d_converged, 0, sizeof(int)); // Should start from zero?

  int num_blocks = args.num_cluster*args.dims/NUMBER_OF_THREADS;
  if (num_blocks == 0) num_blocks = 1;
  cudaEventRecord(start_t);

  d_cuda_convergence_helper<<<dim3(num_blocks), dim3(NUMBER_OF_THREADS)>>>(d_new_centroids, d_old_centroids, d_intermediate_values, args.dims, args.num_cluster);
  cudaEventRecord(stop_t);
  cudaDeviceSynchronize();
  // float total_time = 0;
  // cudaEventElapsedTime(&total_time, start_t, stop_t);
  // std::cout << "cuda_converged: " << total_time << std::endl;
  num_blocks = args.num_cluster/NUMBER_OF_THREADS;
  if (num_blocks == 0) num_blocks = 1;
  cudaEventRecord(start_t);

  d_cuda_convergence_helper_threshold<<<num_blocks, NUMBER_OF_THREADS>>>(d_intermediate_values, d_converged, args.num_cluster, args.threshold);
  cudaEventRecord(stop_t);

  cudaDeviceSynchronize();
//  total_time = 0;
//   cudaEventElapsedTime(&total_time, start_t, stop_t);
//   std::cout << "cuda_converged 2: " << total_time << std::endl;
  //Sync

  // Copy Memory back from Device to Host
  cudaMemcpy(h_converged, d_converged, sizeof(int), cudaMemcpyDeviceToHost);

  bool converged = true;
  if (*h_converged != 0){
    converged = false;
  }

  // Free Device Memory
  cudaFree(d_intermediate_values);
  cudaFree(d_converged);

  // Free Host Memory
  free(h_converged);
  // Check if each of the centroid has moved less than the threshold provided.
  return converged;
}

__global__ void d_cuda_convergence_helper(float * new_c, float * old_c, float * temp, int dimensions, int num_cluster){
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < dimensions * num_cluster){
    atomicAdd(&temp[index/dimensions], (float)powf( new_c[index] - old_c[index], 2.0));
  }
}

__global__ void d_cuda_convergence_helper_threshold(float * temp, int * converged, int num_cluster, float threshold){
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < num_cluster){
    if (threshold < sqrtf(temp[index])){
      atomicAdd(converged, 1);
    }
  }
}

float * cuda_copy(float * original, options_t args)
{
  float * copy = (float *) malloc(args.num_cluster * args.dims * sizeof(float));

  for (int i =0; i < args.num_cluster * args.dims; i++){
    copy[i] = original[i];
  }

  return copy;
}
