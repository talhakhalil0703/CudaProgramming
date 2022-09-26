#include "io.h"
#include <iostream>
#include <fstream>

void read_file(struct options_t* args,
               double***             input_vals,
               double***             output_vals) {

    int dimensions = args->dims;

  	// Open file
	std::ifstream in;
	in.open(args->inputfilename);
    if (in.is_open()){

	// Get num vals
	in >> args->number_of_values;

    double ** points;
    int garbage;
    points = (double **) malloc (args->number_of_values * sizeof(double *));
    double * point;
    for (int i =0; i < args->number_of_values; i++){
        //discard the first value this is the index, starting from 1...
        in >> garbage;
        point = (double *) malloc (dimensions * sizeof(double));
        for (int j =0; j<dimensions; j++){
            in >> point[j];
        }
        points[i] = point;
    }
    *input_vals = points;
    in.close();
    } else{
        std::cout << "Could not open file" << std::endl;
    }
}

void free_input_points(double ** input, int number_of_values){
    for (int i =0;  i < number_of_values; i++){
        free(input[i]);
    }
    free(input);
}

void print_points(double ** input, int number_of_values, int dimensions){
    for (int i =0; i < number_of_values; i++) {
        std::cout << i << ": ";
        for (int j =0; j < dimensions; j++){
            std::cout << input[i][j] << " ";
        }
        std::cout << std::endl;
    }
}