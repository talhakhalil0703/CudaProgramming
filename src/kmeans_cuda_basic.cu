#include "kmeans_cuda_basic.h"
#include "random.h"
#include "io.h"
// #include <cmath>
#include <limits>
#include <math.h>

void kmeans_cuda_basic(double **d_dataset, int clusters, options_t &args) {

  // TODO: Do random assigning of centroids in the main function or helper somewhere else?
  double **d_centroids = (double **)malloc(args.num_cluster * sizeof(double *));
  int index = 0;
  for (int i = 0; i < args.num_cluster; i++){
    index = kmeans_rand() % args.number_of_values;
    d_centroids[i] = d_dataset[index];
  }

  // print_points(centroids, args.num_cluster ,args.dims);
  int iterations = 0;
  double * old_centroids = NULL;
  bool done = false;
  int * labels;


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

  while(!done){
    //copy
    old_centroids = cuda_copy(centroids, args);

    iterations++;

    //labels is a mapping from each point in the dataset to the enarest euclidian distance centroid
    labels = cuda_find_nearest_centroids(dataset, centroids, args);

    // Print Labels
    // for (int i =0 ; i< args.number_of_values; i++){
    //   std::cout << i << ": " << labels[i] << std::endl;
    // }

    //the new centroids are the average of all points that map to each centroid
    centroids = cuda_average_labeled_centroids(dataset, labels, args);

    done = iterations > args.max_num_iter || cuda_converged(centroids, old_centroids, args);

    free(old_centroids);
    // free labels, only if not done
    free (labels);
    printf("Iterations : %d\n", iterations);
    // print_points(centroids, args.num_cluster, args.dims);
  }

  print_points(centroids, args.num_cluster, args.dims);
}

int * cuda_find_nearest_centroids(double * h_dataset, double * h_centroids, options_t &args){

  int * h_labels = (int *)malloc(args.number_of_values * sizeof(int));

  //Allocate Device Memory
  double * d_dataset;
  double * d_centroids;
  double * d_intermediate_values;
  int * d_labels;

  cudaMalloc((void**)&d_dataset, args.dims*args.number_of_values*sizeof(double));
  cudaMalloc((void**)&d_centroids, args.dims*args.num_cluster*sizeof(double));
  cudaMalloc((void**)&d_intermediate_values, args.number_of_values * args.num_cluster * sizeof(double));
  cudaMalloc((void**)&d_labels, args.number_of_values * sizeof(int));

  // Transfer Memory from Host to Device
  cudaMemcpy(d_dataset, h_dataset, args.dims*args.number_of_values*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_centroids, h_centroids, args.dims*args.num_cluster*sizeof(double), cudaMemcpyHostToDevice);

  //Launch the kernel
  d_cuda_find_nearest_centroids<<<dim3(args.number_of_values), dim3(args.num_cluster)>>>(d_dataset, d_centroids, d_intermediate_values, d_labels, args.dims, std::numeric_limits<double>::max());

  //Sync
  cudaDeviceSynchronize();

  // Copy Memory back from Device to Host
  cudaMemcpy(h_labels, d_labels, args.number_of_values*sizeof(int), cudaMemcpyDeviceToHost);

  //Free Device Memory
  cudaFree(d_dataset);
  cudaFree(d_centroids);
  cudaFree(d_intermediate_values);
  cudaFree(d_labels);

  return h_labels;
}

__global__ void d_cuda_find_nearest_centroids(double * dataset, double * centroids, double * temp, int * labels, int dims, double max){
  //For the dataset the thread id should not matter as each thread should point to the same point in the dataset, a block maps to a point
  int point_starting_index = blockIdx.x * dims;

  //Likewise each centroid does not care about what block it's in it only cares about what thread it's in, a thread maps to a centroid
  int thread_start_index = threadIdx.x * dims;

  //Unique index we use to store the distance for each point to each centroid
  int stored_index = threadIdx.x + blockIdx.x * blockDim.x;

  if (threadIdx.x < blockDim.x){
    double distance = 0;
    for (int i = 0; i < dims; i++ ){
      // Centroid indexing is different from the indexing of the data set!
      // This needs to be looked into further, when you look at this next write out pseudo code first
      // My thinking was that each <<<block, threads>>> each block would find the label for each point,
      // Where each thread would find the distance for point vs centroid, the block would sync and choose
      // lowest point to assign the label as.
      // In this case the starting index of the centroids is independent of what block you are in
      distance += powf( dataset[point_starting_index+i] - centroids[thread_start_index + i], 2.0);
    }
  // At this point now each thread has caluclated their own distance, this should now be stored somewhere linked to said thread
  // Threads is mapped to the cluster
  distance = sqrtf(distance);
  temp[stored_index] = distance;
  }

  __syncthreads();

  // Each thread at this point now has calculated the distance, we now should find the centroid with the smallest distance.
  if (threadIdx.x == 0) {
    double shortest_distance = max;
    int id_short = 7;
    // Loop through all the clusters, recall that the number of threads is the number of clusters, and here blockDim.x is number of clusters
    for (int j =0; j < blockDim.x; j++){
      if (temp[stored_index + j] < shortest_distance)
      {
        id_short = j;
        shortest_distance = temp[stored_index + j];
      }
    }

    //A label needs to be only stored per point, each block is given a point thus, each block id represents the points index in the label
    labels[blockIdx.x] = id_short;
  }
}
double * cuda_average_labeled_centroids(double * h_dataset, int * h_labels, options_t &args){
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
  d_cuda_average_labeled_centroids<<<dim3(args.num_cluster), dim3(args.dims)>>>(d_dataset, d_labels, d_centroids, args.number_of_values);

  // Sync
  cudaDeviceSynchronize();
  // Copy Memory back from Device to Host
  cudaMemcpy(h_centroids, d_centroids, args.num_cluster * args.dims * sizeof(double), cudaMemcpyDeviceToHost);
  // Free Device Memory
  cudaFree(d_dataset);
  cudaFree(d_labels);
  cudaFree(d_centroids);

  return h_centroids;
}

__global__ void d_cuda_average_labeled_centroids(double * d_dataset, int * d_labels, double * centroids, int number_of_values){
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


bool cuda_converged(double * h_new_centroids, double* h_old_centroids, options_t &args) {

  bool * h_convergence = (bool *)malloc(args.num_cluster * sizeof(double));

  //Allocate Device Memory
  double * d_new_centroids;
  double * d_old_centroids;
  double * d_intermediate_values;
  bool * d_convergence;

  cudaMalloc((void**)&d_new_centroids, args.dims*args.num_cluster*sizeof(double));
  cudaMalloc((void**)&d_old_centroids, args.dims*args.num_cluster*sizeof(double));
  cudaMalloc((void**)&d_intermediate_values, args.dims*args.num_cluster*sizeof(double));
  cudaMalloc((void**)&d_convergence, args.num_cluster*sizeof(bool));

  // Transfer Memory from Host to Device
  cudaMemcpy(d_new_centroids, h_new_centroids, args.dims*args.num_cluster*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_old_centroids, h_old_centroids, args.dims*args.num_cluster*sizeof(double), cudaMemcpyHostToDevice);

  d_cuda_convergence_helper<<<dim3(args.num_cluster), dim3(args.dims)>>>(d_new_centroids, d_old_centroids, d_intermediate_values, d_convergence, args.threshold, args.dims);

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
  cudaFree(d_intermediate_values);
  cudaFree(d_convergence);

  // Free Host Memory
  free(h_convergence);

  // Check if each of the centroid has moved less than the threshold provided.
  return converged;
}

__global__ void d_cuda_convergence_helper(double * new_c, double * old_c, double * temp, bool * convergence, double threshold, int dimensions){
  int index = threadIdx.x + blockIdx.x * blockDim.x;

  if (threadIdx.x < dimensions){
    temp[index] = powf( new_c[index] - old_c[index], 2.0);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    double distance = 0;
    for (int j =0; j < dimensions; j++){
      distance += temp[blockIdx.x * blockDim.x + j];
    }

    distance = sqrtf(distance);

    if (threshold < distance){
      convergence[blockIdx.x] = false;
    } else {
      convergence[blockIdx.x] = true;
    }
  }
}


double cuda_eucledian_distance(double * h_first, double * h_second, int h_dimensions){
  double sum = 0;

  // Allocate arrays in Host Memory
  double * h_sum = (double *)malloc(sizeof(double));

  // Allocate arrays in Device Memory
  double * d_first;
  double * d_second;
  double * d_return;
  double * d_sum;

  cudaMalloc((void**)&d_first, h_dimensions* sizeof(double));
  cudaMalloc((void**)&d_second, h_dimensions* sizeof(double));
  cudaMalloc((void**)&d_return, h_dimensions* sizeof(double));
  cudaMalloc((void**)&d_sum, sizeof(double));


  // Copy memory from Host to Device
  // Where to, From, How many, Direction
  cudaMemcpy(d_first, h_first, h_dimensions*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_second, h_second, h_dimensions*sizeof(double), cudaMemcpyHostToDevice);

  // Launch Kernel
  d_eucledian_distance_helper<<<dim3(1), dim3(h_dimensions)>>> (d_first, d_second, d_return, d_sum, h_dimensions);

  cudaDeviceSynchronize();

  cudaMemcpy(h_sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost);

  // Clear Device Memory and Host return Memory
  cudaFree(d_first);
  cudaFree(d_second);
  cudaFree(d_return);
  cudaFree(d_sum);

  sum = *h_sum;

  free(h_sum);

  return sum;
}

__global__ void d_eucledian_distance_helper(double * first, double * second, double * pow, double * ret, int dim){

  int i = threadIdx.x;

  if (i < dim){
    pow[i] = powf( (double) first[i] - second[i], (double)2.0);
  }
  __syncthreads();

  if (i == 0){
    double sum =0;
    for (int j = 0; j < dim; j++){
      sum += pow[i];
    }

    *ret = sqrtf(sum);
  }

}


double * cuda_copy(double * original, options_t args)
{
  double * copy = (double *) malloc(args.num_cluster * args.dims * sizeof(double));

  for (int i =0; i < args.num_cluster * args.dims; i++){
    copy[i] = original[i];
  }

  return copy;
}
