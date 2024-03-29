#include "kmeans_cuda_shared.h"
#include "random.h"
#include "io.h"
#include <limits>
#include <chrono>
#include <math.h>

#define NUMBER_OF_THREADS 1024

#ifdef PRINT_TIMES
static float mem_time = 0;
#endif

void kmeans_cuda_shared(float *dataset, float * centroids, options_t &args) {
  int iterations = 0;
  bool done = false;
  float duration_total = 0;
  float duration = 0;

  #ifdef PRINT_TIMES
  cudaEvent_t mem_start, mem_stop;
  cudaEventCreate(&mem_start);
  cudaEventCreate(&mem_stop);
  cudaEventRecord(mem_start);
  #endif

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

  #ifdef PRINT_TIMES
  cudaEventRecord(mem_stop);
  cudaDeviceSynchronize();
  {
    float temp =  0;
    cudaEventElapsedTime(&temp, mem_start, mem_stop);
    mem_time += temp;
  }
  #endif

  cudaEvent_t start_t, stop_t;
  cudaEventCreate(&start_t);
  cudaEventCreate(&stop_t);
  cudaEventRecord(start_t);

  while(!done){
    //copy
    duration = 0;

    #ifdef PRINT_TIMES
    cudaEventRecord(mem_start);
    #endif

    cudaMemcpy(old_centroids, d_centroids, args.num_cluster * args.dims * sizeof(float), cudaMemcpyDeviceToDevice);

    #ifdef PRINT_TIMES
    cudaEventRecord(mem_stop);
    cudaDeviceSynchronize();
    {
      float temp =  0;
      cudaEventElapsedTime(&temp, mem_start, mem_stop);
      mem_time += temp;
    }
    #endif

    iterations++;

    //labels is a mapping from each point in the dataset to the enarest euclidian distance centroid
    cuda_shared_find_nearest_centroids(d_dataset, d_labels, d_centroids, args);

    //the new centroids are the average of all points that map to each centroid
    cuda_shared_average_labeled_centroids(d_centroids, d_dataset, d_labels, args);

    done = iterations > args.max_num_iter || cuda_shared_converged(d_centroids, old_centroids, args);

    duration_total += duration;
  }
  cudaEventRecord(stop_t);
  cudaDeviceSynchronize();

  float total_time = 0;
  cudaEventElapsedTime(&total_time, start_t, stop_t);

  printf("%d,%f\n", iterations, total_time/iterations);

  int * labels;
  labels = (int *) malloc(args.number_of_values*sizeof(int));

  #ifdef PRINT_TIMES
  cudaEventRecord(mem_start);
  #endif

  cudaMemcpy(labels, d_labels, args.number_of_values*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(centroids,d_centroids, args.num_cluster * args.dims * sizeof(float), cudaMemcpyDeviceToHost);

  #ifdef PRINT_TIMES
  cudaEventRecord(mem_stop);
  cudaDeviceSynchronize();
  {
    float temp =  0;
    cudaEventElapsedTime(&temp, mem_start, mem_stop);
    mem_time += temp;
  }
  #endif

  #ifdef PRINT_TIMES
  cudaEventRecord(mem_start);
  #endif

  cudaFree(old_centroids);
  cudaFree(d_labels);
  cudaFree(d_centroids);
  cudaFree(d_dataset);

  #ifdef PRINT_TIMES
  cudaEventRecord(mem_stop);
  cudaDeviceSynchronize();
  {
    float temp =  0;
    cudaEventElapsedTime(&temp, mem_start, mem_stop);
    mem_time += temp;
  }
  #endif

  #ifdef PRINT_TIMES
  printf("Time spent copying memory: %lf\n", mem_time);
  #endif

  args.labels = labels;
  args.centroids = centroids;
}

void cuda_shared_find_nearest_centroids(float * d_dataset, int * d_labels, float * d_centroids, options_t &args){

  #ifdef PRINT_TIMES
  //Launch the kernel
  cudaEvent_t start_t, stop_t;
  cudaEventCreate(&start_t);
  cudaEventCreate(&stop_t);
  #endif

  int num_blocks = args.number_of_values/NUMBER_OF_THREADS;
  if (num_blocks == 0) num_blocks = 1;

  #ifdef PRINT_TIMES
  cudaEventRecord(start_t);
  #endif

  d_cuda_shared_find_nearest_centroids<<<dim3(num_blocks), dim3(NUMBER_OF_THREADS), args.dims*args.num_cluster*sizeof(float)>>>(d_dataset, d_centroids, d_labels, args.dims, args.num_cluster, std::numeric_limits<float>::max(), args.number_of_values);
  //Sync

  #ifdef PRINT_TIMES
  cudaEventRecord(stop_t);
  #endif

  cudaDeviceSynchronize();

  #ifdef PRINT_TIMES
    float total_time = 0;
    cudaEventElapsedTime(&total_time, start_t, stop_t);
    std::cout << "find_nearest_centroids: " << total_time << std::endl;
  #endif

}


__global__ void d_cuda_shared_find_nearest_centroids(float * dataset, float * centroids, int * labels, int dims, int num_cluster, float max, int number_of_values){
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  // Centroids all can be copied willy-nilly
  extern __shared__ float s_centroids[];
  // I don't see a point where dims*num_cluster > NUMBER_OF_THREADS, if that happens don't so shared
  if (threadIdx.x < dims*num_cluster){
    s_centroids[threadIdx.x] = centroids[threadIdx.x];
  }

  __syncthreads();
  //Labels don't have to be copied here because they are only accessed once for writing

  // Each thread is given a point and for each point we want to find the closest centroid.

  // Ensure that the point is actually a point that exists
  if (index <  number_of_values){
    float shortest_distance = max;
    float current_distance = 0;
    int closest_index = 0;

    for (int i = 0; i < num_cluster; i++){
      current_distance = 0;
      for (int j =0 ; j < dims; j++){
        current_distance += (dataset[index*dims + j] - s_centroids[i*dims + j]) * (dataset[index*dims + j] - s_centroids[i*dims + j]);
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

void cuda_shared_average_labeled_centroids(float * d_centroids, float * d_dataset, int * d_labels, options_t &args){
  // Allocate Device Memory
  #ifdef PRINT_TIMES
  cudaEvent_t start_t, stop_t;
  cudaEventCreate(&start_t);
  cudaEventCreate(&stop_t);

  cudaEvent_t mem_start, mem_stop;
  cudaEventCreate(&mem_start);
  cudaEventCreate(&mem_stop);
  #endif

  #ifdef PRINT_TIMES
  cudaEventRecord(mem_start);
  #endif
  int * d_points_in_centroids;
  cudaMalloc ((void **)&d_points_in_centroids, args.num_cluster*sizeof(int));

  // Transfer Memory From Host To Device
  
  #ifdef PRINT_TIMES
  cudaEventRecord(mem_stop);
  cudaDeviceSynchronize();
  {
    float temp =  0;
    cudaEventElapsedTime(&temp, mem_start, mem_stop);
    mem_time += temp;
  }
  #endif

  #ifdef PRINT_TIMES
  cudaEventRecord(mem_start);
  #endif

  cudaMemset(d_points_in_centroids, 0, args.num_cluster * sizeof(int)); // Should start from zero?
  cudaMemset(d_centroids, 0, args.num_cluster * args.dims * sizeof(float)); // Should start from zero?

  #ifdef PRINT_TIMES
  cudaEventRecord(mem_stop);
  cudaDeviceSynchronize();
  {
    float temp =  0;
    cudaEventElapsedTime(&temp, mem_start, mem_stop);
    mem_time += temp;
  }
  #endif

  // Launch the kernel
  int num_blocks = args.number_of_values/NUMBER_OF_THREADS;
  if (num_blocks == 0) num_blocks = 1;

  #ifdef PRINT_TIMES
  cudaEventRecord(start_t);
  #endif

  d_cuda_shared_average_labeled_centroids<<<dim3(num_blocks), dim3(NUMBER_OF_THREADS), args.dims*args.num_cluster*sizeof(float)>>>(d_dataset, d_labels, d_points_in_centroids, d_centroids, args.number_of_values, args.dims, args.num_cluster);

  // Sync
  #ifdef PRINT_TIMES
  cudaEventRecord(stop_t);
  #endif

  cudaDeviceSynchronize();
  #ifdef PRINT_TIMES
  float total_time = 0;
  cudaEventElapsedTime(&total_time, start_t, stop_t);
  std::cout << "averaged_labeled_centroids: " << total_time << std::endl;
  #endif
  num_blocks = args.num_cluster*args.dims/NUMBER_OF_THREADS;
  if (num_blocks == 0) num_blocks = 1;

  #ifdef PRINT_TIMES
  cudaEventRecord(start_t);
  #endif

  d_cuda_shared_average_labeled_centroids_divide<<<num_blocks,NUMBER_OF_THREADS>>>(d_centroids, d_points_in_centroids, args.dims, args.num_cluster);
  // Sync

  #ifdef PRINT_TIMES
  cudaEventRecord(stop_t);
  #endif

  cudaDeviceSynchronize();
  #ifdef PRINT_TIMES
  total_time = 0;
  cudaEventElapsedTime(&total_time, start_t, stop_t);
  std::cout << "averaged_labeled_centroids 3: " << total_time << std::endl;
  #endif

  #ifdef PRINT_TIMES
  cudaEventRecord(mem_start);
  #endif

  // Free Device Memory
  cudaFree(d_points_in_centroids);

  #ifdef PRINT_TIMES
  cudaEventRecord(mem_stop);
  cudaDeviceSynchronize();
  {
    float temp =  0;
    cudaEventElapsedTime(&temp, mem_start, mem_stop);
    mem_time += temp;
  }
  #endif

}

__global__ void d_cuda_shared_average_labeled_centroids(float * dataset, int * labels, int * points_in_centroids, float * centroids, int number_of_values, int dims, int num_cluster){
  // Unique index for each point
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  // Labels is read once no real performan to be gained here, centroids could be....
  // Centroids all can be copied willy-nilly
  extern __shared__ float s_centroids[];
  // I don't see a point where dims*num_cluster > NUMBER_OF_THREADS, if that happens don't do shared
  //Check to ensure that the index is actually less than the total number of values we have
  if (index <  number_of_values) {
    // At the start zero the shared memory, just as the global memory has been
    if (threadIdx.x < dims*num_cluster){
      s_centroids[threadIdx.x] = 0;
    }

    __syncthreads();

    //Get the centroid that the point belongs to.
    int centroid_index = labels[index];

    //Increment the count of points that centroid contains.
    atomicAdd(&points_in_centroids[centroid_index], 1);

    //Increment that centroids dims with that point
    for (int i = 0 ; i < dims;  i++){
      atomicAdd(&s_centroids[centroid_index*dims + i], dataset[index*dims+i]);
    }

    __syncthreads();

    // I need to write the memory back
    if (threadIdx.x < dims*num_cluster){
      atomicAdd(&centroids[threadIdx.x], s_centroids[threadIdx.x] );
    }
  }
}

__global__ void d_cuda_shared_average_labeled_centroids_divide(float * centroids, int * points_in_centroids, int dims, int num_cluster){
  int index = threadIdx.x + blockIdx.x * blockDim.x;
   if (index < dims*num_cluster) {
    //For each centroid divide it's dimensions with the amount of points present.
    //We can store this value
    int points_in_centroid = points_in_centroids[index/dims];

    centroids[index] /= points_in_centroid;
   }
}

bool cuda_shared_converged(float * d_new_centroids, float* d_old_centroids, options_t &args) {

  int * h_converged = (int *) malloc(sizeof(int));

  // Allocate Device Memory
  #ifdef PRINT_TIMES
  cudaEvent_t start_t, stop_t;
  cudaEventCreate(&start_t);
  cudaEventCreate(&stop_t);
  cudaEvent_t mem_start, mem_stop;
  cudaEventCreate(&mem_start);
  cudaEventCreate(&mem_stop);
  #endif

  //Allocate Device Memory
  float * d_intermediate_values;
  int * d_converged;
  #ifdef PRINT_TIMES
  cudaEventRecord(mem_start);
  #endif
  
  cudaMalloc((void**)&d_intermediate_values, args.num_cluster*sizeof(float));
  cudaMalloc((void**)&d_converged, sizeof(int));

  #ifdef PRINT_TIMES
  cudaEventRecord(mem_stop);
  cudaDeviceSynchronize();
  {
    float temp =  0;
    cudaEventElapsedTime(&temp, mem_start, mem_stop);
    mem_time += temp;
  }
  #endif

  #ifdef PRINT_TIMES
  cudaEventRecord(mem_start);
  #endif
  // Transfer Memory from Host to Device
  cudaMemset(d_intermediate_values, 0, args.num_cluster * sizeof(float)); // Should start from zero?
  cudaMemset(d_converged, 0, sizeof(int)); // Should start from zero?

  #ifdef PRINT_TIMES
  cudaEventRecord(mem_stop);
  cudaDeviceSynchronize();
  {
    float temp =  0;
    cudaEventElapsedTime(&temp, mem_start, mem_stop);
    mem_time += temp;
  }
  #endif

  int num_blocks = args.num_cluster*args.dims/NUMBER_OF_THREADS;
  if (num_blocks == 0) num_blocks = 1;

  #ifdef PRINT_TIMES
  cudaEventRecord(start_t);
  #endif

  d_cuda_shared_convergence_helper<<<dim3(num_blocks), dim3(NUMBER_OF_THREADS)>>>(d_new_centroids, d_old_centroids, d_intermediate_values, args.dims, args.num_cluster);

  #ifdef PRINT_TIMES
  cudaEventRecord(stop_t);
  #endif

  cudaDeviceSynchronize();

  #ifdef PRINT_TIMES
  float total_time = 0;
  cudaEventElapsedTime(&total_time, start_t, stop_t);
  std::cout << "cuda_converged: " << total_time << std::endl;
  num_blocks = args.num_cluster/NUMBER_OF_THREADS;
  #endif
  if (num_blocks == 0) num_blocks = 1;

  #ifdef PRINT_TIMES
  cudaEventRecord(start_t);
  #endif

  d_cuda_shared_convergence_helper_threshold<<<num_blocks, NUMBER_OF_THREADS>>>(d_intermediate_values, d_converged, args.num_cluster, args.threshold);

  #ifdef PRINT_TIMES
  cudaEventRecord(stop_t);
  #endif

  cudaDeviceSynchronize();
  #ifdef PRINT_TIMES

  total_time = 0;
  cudaEventElapsedTime(&total_time, start_t, stop_t);
  std::cout << "cuda_converged 2: " << total_time << std::endl;
  #endif


  #ifdef PRINT_TIMES
  cudaEventRecord(mem_start);
  #endif
  // Copy Memory back from Device to Host

  cudaMemcpy(h_converged, d_converged, sizeof(int), cudaMemcpyDeviceToHost);

  #ifdef PRINT_TIMES
  cudaEventRecord(mem_stop);
  cudaDeviceSynchronize();
  {
    float temp =  0;
    cudaEventElapsedTime(&temp, mem_start, mem_stop);
    mem_time += temp;
  }
  #endif

  bool converged = true;
  if (*h_converged != 0){
    converged = false;
  }

  #ifdef PRINT_TIMES
  cudaEventRecord(mem_start);
  #endif
  
  // Free Device Memory
  cudaFree(d_intermediate_values);
  cudaFree(d_converged);

  // Free Host Memory
  free(h_converged);
  
  #ifdef PRINT_TIMES
  cudaEventRecord(mem_stop);
  cudaDeviceSynchronize();
  {
    float temp =  0;
    cudaEventElapsedTime(&temp, mem_start, mem_stop);
    mem_time += temp;
  }
  #endif

  // Check if each of the centroid has moved less than the threshold provided.
  return converged;
}

__global__ void d_cuda_shared_convergence_helper(float * new_c, float * old_c, float * temp, int dimensions, int num_cluster){
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < dimensions * num_cluster){
    atomicAdd(&temp[index/dimensions], (new_c[index] - old_c[index])*(new_c[index] - old_c[index]));
  }
}

__global__ void d_cuda_shared_convergence_helper_threshold(float * temp, int * converged, int num_cluster, float threshold){
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  __shared__ int s_converged;
  if (index == 0){
    s_converged = 0;
  }
  __syncthreads();
  if (index < num_cluster){
    if (threshold < sqrtf(temp[index])){
      atomicAdd(&s_converged, 1);
    }
  }
  __syncthreads();
  if (index == 0 && s_converged > 0){
    *converged = s_converged;
  }

}