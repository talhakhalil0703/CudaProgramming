#include "io.h"
#include <iostream>
#include <fstream>
#include <iomanip>

void read_file(struct options_t* args,
               float**             input_vals) {

    int dimensions = args->dims;

  	// Open file
	std::ifstream in;
	in.open(args->inputfilename);
    if (in.is_open()){

	// Get num vals
	in >> args->number_of_values;

    int garbage;
    float * points = (float *) malloc (args->number_of_values * args->dims * sizeof(float));
    int index = 0;
    for (int i =0; i < args->number_of_values; i++){
        //discard the first value this is the index, starting from 1...
        in >> garbage;
        for (int j =0; j<dimensions; j++){
            in >> points[index++];
        }
    }

    *input_vals = points;
    in.close();
    } else{
        std::cout << "Could not open file" << std::endl;
    }
}
