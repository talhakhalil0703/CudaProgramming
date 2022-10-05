#include "kmeans_thrust.h"
#include "random.h"
#include "io.h"
#include <limits>
#include <chrono>
#include <math.h>
#include <thrust/for_each.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/reduce.h>
#include <thrust/tuple.h>
#include <thrust/sort.h>

typedef thrust::device_vector<float>::iterator ElementIterator;
typedef thrust::device_vector<int>::iterator   IndexIterator;
typedef thrust::tuple<float, int> FloatIntTuple;

struct printf_functor
{
  __host__ __device__
  void operator()(float x)
  {
    // note that using printf in a __device__ function requires
    // code compiled for a GPU with compute capability 2.0 or
    // higher (nvcc --arch=sm_20)
    printf("%f\n", x);
  }
};

struct divide_by_count : public thrust::unary_function<int, int>{
  int div;
  __host__ __device__
  divide_by_count(int x) : div(x) {}

  __host__ __device__
  int operator()(int input){
      return input / div;
  }
};

struct mod_by_count : public thrust::unary_function<int, int>{
  int mod;
  __host__ __device__
  mod_by_count(int x) : mod(x) {}

  __host__ __device__
  int operator()(int input){
      return input % mod;
  }
};

struct square_root : public thrust::unary_function<float, float>{
  __host__ __device__
  float operator()(float input){
      return sqrtf(input);
  }
};

struct square_difference : thrust::binary_function<float, float, float>
{
    __host__ __device__
  float operator()(float a, float b)
  {
    // note that using printf in a __device__ function requires
    // code compiled for a GPU with compute capability 2.0 or
    // higher (nvcc --arch=sm_20)
    return (a-b)*(a-b);
  }
};

struct less_than_key: thrust::binary_function<FloatIntTuple, FloatIntTuple, bool>
{
    __host__ __device__
    inline bool operator() (FloatIntTuple struct1,  FloatIntTuple struct2)
    {
      return thrust::get<0>(struct1) <  thrust::get<0>(struct2);
    }
};

struct get_min_label : thrust::binary_function<FloatIntTuple, FloatIntTuple, FloatIntTuple>
{
    __host__ __device__
    FloatIntTuple operator()(FloatIntTuple a, FloatIntTuple b)
  {
    // note that using printf in a __device__ function requires
    // code compiled for a GPU with compute capability 2.0 or
    // higher (nvcc --arch=sm_20)
    float a_val = (thrust::get<0>(a));
    float b_val = (thrust::get<0>(b));
    if (a_val < b_val) {
      return a;
    } else {
      return b;
    }
  }
};

#define D_VFLOAT thrust::device_vector<float>
#define D_VINT thrust::device_vector<int>
#define PRINT


void kmeans_thrust(float * dataset, float * centroids, options_t &args) {
  // Host to device
  D_VFLOAT d_dataset(args.number_of_values*args.dims);
  D_VFLOAT d_centroids(args.num_cluster*args.dims);

  for (int i =0; i < args.number_of_values*args.dims; i++){
    d_dataset[i] = dataset[i];
  }

  for (int i =0; i < args.num_cluster*args.dims; i++){
    d_centroids[i] = centroids[i];
  }

  #ifdef PRINT
  std::cout << "Dataset: " << std::endl;
  for(int i =0; i < args.number_of_values; i++){
    std::cout << i << " : ";
    for (int j =0; j < args.dims; j++){
      std::cout << d_dataset[i*args.dims + j] << " ";
    }
    std::cout << std::endl;

  }
  #endif

  #ifdef PRINT
  std::cout << "Centroids: " << std::endl;
  for(int i =0; i < args.num_cluster; i++){
    std::cout << i << " : ";
    for (int j =0; j < args.dims; j++){
      std::cout << d_centroids[i*args.dims + j] << " ";
    }
    std::cout << std::endl;

  }
  #endif

  D_VINT d_centroids_value_times_indices(args.num_cluster*args.dims*args.number_of_values);
  thrust::sequence(thrust::device, d_centroids_value_times_indices.begin(), d_centroids_value_times_indices.end(), 0);
  thrust::transform(thrust::device, d_centroids_value_times_indices.begin(), d_centroids_value_times_indices.end(), d_centroids_value_times_indices.begin(), mod_by_count(args.number_of_values*args.num_cluster)); // 0 1 2 0 1 2 0 1 2
  thrust::permutation_iterator<ElementIterator, IndexIterator> d_centroids_value_times(d_centroids.begin(), d_centroids_value_times_indices.begin());

  D_VINT d_dataset_value_times_indices(args.num_cluster*args.dims*args.number_of_values);
  thrust::sequence(thrust::device, d_dataset_value_times_indices.begin(), d_dataset_value_times_indices.end(), 0);
  thrust::transform(thrust::device, d_dataset_value_times_indices.begin(), d_dataset_value_times_indices.end(), d_dataset_value_times_indices.begin(), mod_by_count(args.number_of_values*args.dims)); // 0 0 1 1 2 2
  thrust::permutation_iterator<ElementIterator, IndexIterator> d_dataset_value_times(d_dataset.begin(), d_dataset_value_times_indices.begin());

  D_VFLOAT d_squared_difference(args.num_cluster*args.dims*args.number_of_values);

  // Take a transformation between the two vectors, subtract and square them
  thrust::transform(thrust::device, d_dataset_value_times, d_dataset_value_times + args.num_cluster*args.dims*args.number_of_values , d_centroids_value_times, d_squared_difference.begin(), square_difference());


  #ifdef PRINT
  std::cout << "Squared Distances In General: " << std::endl;
  for(int i =0; i < args.num_cluster; i++){
    for (int j =0; j < args.number_of_values; j++){
      for (int k =0; k < args.dims; k++){
        int index = i * args.number_of_values * args.dims + j * args.dims + k;
      std::cout << "index: " << index << " centroid: " << d_centroids_value_times_indices[index] << " dataset: "  << d_dataset_value_times_indices[index] << " " << d_squared_difference[index] << std::endl;
      }
    }
  }
  std::cout << std::endl;
  #endif

  //Reduce the resulting vector by keys, map of keys is repeating every dimension so that we have a  num_cluster*num_val distances
  D_VINT d_reduction_indices(args.num_cluster*args.dims*args.number_of_values);
  D_VINT d_reduced_indices(args.num_cluster*args.number_of_values);
  D_VFLOAT d_reduced_needs_squaring(args.num_cluster*args.number_of_values);
  D_VFLOAT d_reduced_squared(args.num_cluster*args.number_of_values);
  thrust::sequence(thrust::device, d_reduction_indices.begin(), d_reduction_indices.end(), 0);
  thrust::transform(thrust::device, d_reduction_indices.begin(), d_reduction_indices.end(), d_reduction_indices.begin(), divide_by_count(args.dims));
  thrust::reduce_by_key(d_reduction_indices.begin(), d_reduction_indices.end(), d_squared_difference.begin(), d_reduced_indices.begin(), d_reduced_needs_squaring.begin());
  thrust::transform(thrust::device, d_reduced_needs_squaring.begin(), d_reduced_needs_squaring.end(), d_reduced_squared.begin(), square_root());

  D_VINT d_new_indices(args.number_of_values*args.num_cluster);
  thrust::sequence(thrust::device, d_new_indices.begin(), d_new_indices.end(), 0);
  thrust::transform(thrust::device, d_new_indices.begin(), d_new_indices.end(), d_new_indices.begin(), divide_by_count(args.num_cluster));

  D_VINT d_indices_for_labels(args.number_of_values*args.num_cluster);
  thrust::sequence(thrust::device, d_indices_for_labels.begin(), d_indices_for_labels.end(), 0);
  thrust::transform(thrust::device, d_indices_for_labels.begin(), d_indices_for_labels.end(), d_indices_for_labels.begin(), mod_by_count(args.num_cluster));

  #ifdef PRINT
  std::cout << "Distances In General: " << std::endl;
  for (int j =0; j < args.number_of_values; j++){
    std::cout << j << " : ";
    for(int i =0; i < args.num_cluster; i++){
      int index = j*args.num_cluster + i;
      std::cout << "Recuding index: " << d_new_indices[index] << " Centroid Label: " << d_indices_for_labels[index] << " Distance: "<<d_reduced_squared[index] << ", ";
    }
    std::cout << std::endl;
  }
  #endif

  // At this point I now have distances for centroids vs points, how do I got about getting labels
  // Now we can attach the distances with their indices using the zip function
  // Recall that we already know what centroids are where,
  // We want the a [Distances for all centroids point 0] ... [Distances for all centroids point N]
  D_VINT d_temp_keys(args.number_of_values*args.num_cluster);
  D_VINT d_labels(args.number_of_values);
  D_VFLOAT d_temp_distance(args.number_of_values);
  thrust::sequence(thrust::device, d_temp_keys.begin(), d_temp_keys.end(), 0);
  thrust::transform(thrust::device, d_temp_keys.begin(), d_temp_keys.end(), d_temp_keys.begin(), divide_by_count(args.num_cluster));

  thrust::sort_by_key(d_temp_keys.begin(), d_temp_keys.end(), thrust::make_zip_iterator(thrust::make_tuple(d_reduced_squared.begin(),d_indices_for_labels.begin())), less_than_key());

  thrust::reduce_by_key(d_temp_keys.begin(), d_temp_keys.end(), thrust::make_zip_iterator(thrust::make_tuple(d_reduced_squared.begin(),d_indices_for_labels.begin())), d_temp_keys.begin(), thrust::make_zip_iterator(thrust::make_tuple(d_temp_distance.begin(), d_labels.begin())), thrust::equal_to<int>(), get_min_label());

  #ifdef PRINT
  std::cout << "Labels: " << std::endl;
  for(int i =0; i < args.number_of_values; i++){
    std::cout << i << " : "<< d_labels[i] << std::endl;
  }
  #endif

  #ifdef PRINT
  std::cout << "Min Distances: " << std::endl;
  for(int i =0; i < args.number_of_values; i++){
    std::cout << i << " : "<< d_temp_distance[i] << std::endl;
  }
  #endif

  return;

}
