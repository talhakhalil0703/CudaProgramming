#include "kmeans_cuda_shared.h"
#include "random.h"
#include "io.h"
// #include <cmath>
#include <limits>
#include <chrono>
#include <math.h>

__device__ double shared_atomicMin_d(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = atomicMin(address_as_ull, __double_as_longlong(val));
        old = atomicCAS(address_as_ull, old, assumed);

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}

void kmeans_cuda_shared(double * dataset, double * centroids, options_t &args) {

  int iterations = 0;
  double * old_centroids = NULL;
  bool done = false;
  int * labels;
  double duration_total = 0;
  double duration = 0;

  while(!done){
    //copy
    duration = 0;

    old_centroids = cuda_shared_copy(centroids, args);

    iterations++;

    //labels is a mapping from each point in the dataset to the enarest euclidian distance centroid
    labels = cuda_shared_find_nearest_centroids(dataset, centroids, args, &duration);

    // Print Labels
    // for (int i =0 ; i< args.number_of_values; i++){
    //   std::cout << i << ": " << labels[i] << std::endl;
    // }

    //the new centroids are the average of all points that map to each centroid
    centroids = cuda_shared_average_labeled_centroids(dataset, labels, args, &duration);

    done = iterations > args.max_num_iter || cuda_shared_converged(centroids, old_centroids, args, &duration);

    duration_total += duration;
    free(old_centroids);
    // free labels, only if not done
    if (!done) free (labels);
  }

  printf("%d,%lf\n", iterations, duration_total/iterations);

  args.labels = labels;
  args.centroids = centroids;
}

int * cuda_shared_find_nearest_centroids(double * h_dataset, double * h_centroids, options_t &args, double * duration){
  //Timing
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  int * h_labels = (int *)malloc(args.number_of_values * sizeof(int));

  //Allocate Device Memory
  double * d_dataset;
  double * d_centroids;
  int * d_labels;

  cudaMalloc((void**)&d_dataset, args.dims*args.number_of_values*sizeof(double));
  cudaMalloc((void**)&d_centroids, args.dims*args.num_cluster*sizeof(double));
  cudaMalloc((void**)&d_labels, args.number_of_values * sizeof(int));

  // Transfer Memory from Host to Device
  cudaMemcpy(d_dataset, h_dataset, args.dims*args.number_of_values*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_centroids, h_centroids, args.dims*args.num_cluster*sizeof(double), cudaMemcpyHostToDevice);

  //Launch the kernel
  cudaEventRecord(start);
  d_cuda_shared_find_nearest_centroids<<<dim3(args.number_of_values), dim3(args.num_cluster)>>>(d_dataset, d_centroids, d_labels, args.dims, std::numeric_limits<double>::max());
  cudaEventRecord(stop);

  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);
  *duration += ms;

  //Sync
  cudaDeviceSynchronize();

  // Copy Memory back from Device to Host
  cudaMemcpy(h_labels, d_labels, args.number_of_values*sizeof(int), cudaMemcpyDeviceToHost);

  //Free Device Memory
  cudaFree(d_dataset);
  cudaFree(d_centroids);
  cudaFree(d_labels);

  return h_labels;
}

__global__ void d_cuda_shared_find_nearest_centroids(double * dataset, double * centroids, int * labels, int dims, double max){
  __shared__ double s_distance;
  s_distance = max;


  __syncthreads();

  if (threadIdx.x < blockDim.x){
    double distance = 0;
    for (int i = 0; i < dims; i++ ){
      // Centroid indexing is different from the indexing of the data set!
      // This needs to be looked into further, when you look at this next write out pseudo code first
      // My thinking was that each <<<block, threads>>> each block would find the label for each point,
      // Where each thread would find the distance for point vs centroid, the block would sync and choose
      // lowest point to assign the label as.
      // In this case the starting index of the centroids is independent of what block you are in
      distance += powf( dataset[blockIdx.x * dims + i] - centroids[threadIdx.x * dims + i], 2.0);
    }
    // At this point now each thread has caluclated their own distance, this should now be stored somewhere linked to said thread
    // Threads is mapped to the cluster
    distance = sqrtf(distance);
    shared_atomicMin_d(&s_distance, distance);
    __syncthreads();

    if (distance == s_distance){
      labels[blockIdx.x] = threadIdx.x;
    }
  }
}
double * cuda_shared_average_labeled_centroids(double * h_dataset, int * h_labels, options_t &args, double * duration){
  //Timing
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // First turn the dataset into a singular dimension
  double * h_centroids = (double *)malloc(args.num_cluster * args.dims * sizeof(double));

  // Allocate Device Memory
  double * d_dataset;
  int * d_labels;
  double * d_centroids;
  cudaMalloc((void**)&d_dataset, args.number_of_values * args.dims * sizeof(double));
  cudaMalloc((void**)&d_labels, args.number_of_values * sizeof(int));
  cudaMalloc((void**)&d_centroids, args.num_cluster * args.dims * sizeof(double));

  // Transfer Memory From Host To Device
  cudaMemcpy(d_dataset, h_dataset, args.number_of_values * args.dims * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_labels, h_labels, args.number_of_values * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_centroids, 0, args.num_cluster * args.dims * sizeof(double), cudaMemcpyHostToDevice); // Should start from zero?

  // Launch the kernel
  cudaEventRecord(start);
  d_cuda_shared_average_labeled_centroids<<<dim3(args.num_cluster), dim3(args.dims)>>>(d_dataset, d_labels, d_centroids, args.number_of_values);
  cudaEventRecord(stop);

  // Sync
  cudaDeviceSynchronize();
  // Copy Memory back from Device to Host
  cudaMemcpy(h_centroids, d_centroids, args.num_cluster * args.dims * sizeof(double), cudaMemcpyDeviceToHost);
  // Free Device Memory
  cudaFree(d_dataset);
  cudaFree(d_labels);
  cudaFree(d_centroids);

  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);
  *duration += ms;
  return h_centroids;
}

__global__ void d_cuda_shared_average_labeled_centroids(double * d_dataset, int * d_labels, double * centroids, int number_of_values){
  // Dimensions is blockDim.x
  // A block here manages the centroid Id
  // A thread here manages the addition it needs to do for that dimension
  int points = 0;
  // First loop through  d_dataset skipping dim[blockDim.x] times, and check if the value here is equal to our block id
  for (int i = 0; i < number_of_values; i ++) {
    if (d_labels[i] == blockIdx.x) {
      points++;
      centroids[blockIdx.x * blockDim.x + threadIdx.x] += d_dataset[i * blockDim.x + threadIdx.x];
    }
  }

  if (points != 0){
    centroids[blockIdx.x * blockDim.x + threadIdx.x] /= points;
  }

  //Once you have done the addition for all

}


bool cuda_shared_converged(double * h_new_centroids, double* h_old_centroids, options_t &args, double * duration) {

  //Timing
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  bool * h_convergence = (bool *)malloc(args.num_cluster * sizeof(double));

  //Allocate Device Memory
  double * d_new_centroids;
  double * d_old_centroids;
  bool * d_convergence;

  cudaMalloc((void**)&d_new_centroids, args.dims*args.num_cluster*sizeof(double));
  cudaMalloc((void**)&d_old_centroids, args.dims*args.num_cluster*sizeof(double));
  cudaMalloc((void**)&d_convergence, args.num_cluster*sizeof(bool));

  // Transfer Memory from Host to Device
  cudaMemcpy(d_new_centroids, h_new_centroids, args.dims*args.num_cluster*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_old_centroids, h_old_centroids, args.dims*args.num_cluster*sizeof(double), cudaMemcpyHostToDevice);

  cudaEventRecord(start);
  d_cuda_shared_convergence_helper<<<dim3(args.num_cluster), dim3(args.dims)>>>(d_new_centroids, d_old_centroids, d_convergence, args.threshold, args.dims);
  cudaEventRecord(stop);

  //Sync
  cudaDeviceSynchronize();

  // Copy Memory back from Device to Host
  cudaMemcpy(h_convergence, d_convergence, args.num_cluster*sizeof(bool), cudaMemcpyDeviceToHost);

  bool converged = true;

  for (int i =0; i < args.num_cluster; i++){
    if (!h_convergence[i]) {
      converged = false;
      break;
    }
  }

  // Free Device Memory
  cudaFree(d_new_centroids);
  cudaFree(d_old_centroids);
  cudaFree(d_convergence);

  // Free Host Memory
  free(h_convergence);
  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);
  *duration += ms;
  // Check if each of the centroid has moved less than the threshold provided.
  return converged;
}

__global__ void d_cuda_shared_convergence_helper(double * new_c, double * old_c, bool * convergence, double threshold, int dimensions){
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  __shared__ double distance;
  distance = 0;

  if (threadIdx.x < dimensions){
    atomicAdd(&distance, (double)powf( new_c[index] - old_c[index], 2.0));
  }

  __syncthreads();

  // It looks like here maybe we could make use of __atomic_add, would that make a speedup? Not noticeable enough

  if (threadIdx.x == 0) {
    if (threshold < sqrtf(distance)){
      convergence[blockIdx.x] = false;
    } else {
      convergence[blockIdx.x] = true;
    }
  }
}

double * cuda_shared_copy(double * original, options_t args)
{
  double * copy = (double *) malloc(args.num_cluster * args.dims * sizeof(double));

  for (int i =0; i < args.num_cluster * args.dims; i++){
    copy[i] = original[i];
  }

  return copy;
}
