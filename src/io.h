#include "argparse.h"

void read_file(struct options_t* args,
               double***             input_vals,
               double***             output_vals);

void free_input_points(double ** input, int number_of_values);

void print_points(double * input, int number_of_values, int dimensions);
void print_points(double ** input, int number_of_values, int dimensions);