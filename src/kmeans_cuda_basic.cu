#include "kmeans_cuda_basic.h"
#include "random.h"
#include "io.h"
// #include <cmath>
#include <limits>
#include <math.h>

void kmeans_cuda_basic(double **dataset, int clusters, options_t &args) {

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
    old_centroids = cuda_copy_double(centroids, args);

    iterations++;

    //labels is a mapping from each point in the dataset to the enarest euclidian distance centroid
    labels = cuda_find_nearest_centroids(dataset, centroids, args);

    // Print Labels
    // for (int i =0 ; i< args.number_of_values; i++){
    //   std::cout << i << ": " << labels[i] << std::endl;
    // }

    //the new centroids are the average of all points that map to each centroid
    centroids = cuda_average_labeled_centroids(dataset, labels, clusters, args);

    done = iterations > args.max_num_iter || cuda_converged(centroids, old_centroids, args);
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

int * cuda_find_nearest_centroids(double ** dataset, double ** centroids, options_t &args){
  // TODO: This conversion should be done before somewhere else?

  double * h_dataset = (double *)malloc(args.dims * args.number_of_values * sizeof(double));
  double * h_centroids = (double *)malloc(args.dims * args.num_cluster * sizeof(double));
  int * h_labels = (int *)malloc(args.number_of_values * sizeof(int));

  int index =0;
  for (int i = 0; i < args.number_of_values; i++){
    for (int j = 0; j < args.dims; j++){
      h_dataset[index++] = dataset[i][j];
    }
  }

  index = 0;
  for (int i = 0; i < args.num_cluster; i++){
    for (int j = 0; j < args.dims; j++){
      h_centroids[index++] = centroids[i][j];
    }
  }

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

  //Free Host Memory
  free(h_dataset);
  free(h_centroids);


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

double ** cuda_average_labeled_centroids(double ** dataset, int * labels, int clusters, options_t &args){
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

bool cuda_converged(double ** new_centroids, double** old_centroids, options_t &args) {


  // TODO Does data structure have to be rethought here so we don't do this conversion each time?
  // Make array singular dimension

  double * h_new_centroids = (double *)malloc(args.dims * args.num_cluster * sizeof(double));
  double * h_old_centroids = (double *)malloc(args.dims * args.num_cluster * sizeof(double));
  bool * h_convergence = (bool *)malloc(args.num_cluster * sizeof(double));

  int index =0;

  for (int i = 0; i < args.num_cluster; i++){
    for (int j =0; j < args.dims; j++){
      h_new_centroids[index] = new_centroids[i][j];
      h_old_centroids[index] = old_centroids[i][j];
      index++;
    }
  }

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
  free(h_new_centroids);
  free(h_old_centroids);
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


double ** cuda_copy_double(double ** original, options_t args)
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
