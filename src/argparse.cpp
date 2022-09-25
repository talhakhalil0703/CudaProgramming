#include "argparse.h"

void get_opts(int argc,
              char **argv,
              struct options_t *opts)
{
    if (argc == 1)
    {
        std::cout << "Usage:" << std::endl;
        std::cout << "\t--num_cluster or -k <number of clusters>" << std::endl;
        std::cout << "\t--dims or -d <dimenisons of the points>" << std::endl;
        std::cout << "\t--inputfilename or -i <file_path>" << std::endl;
        std::cout << "\t--max_num_iter or -m <number of maximum iterations>" << std::endl;
        std::cout << "\t--threshold or -t <double specifying the threshold for convergence test>" << std::endl;
        std::cout << "\t--seed or -s <integer seed used to specify the seed for rand()>" << std::endl;
        std::cout << "\t[Optional] --output_conrol or -c flag that controls output of the program" << std::endl;
        std::cout << "\tOnly provide one of the following:" << std::endl;
        std::cout << "\t\t[Optional] --use_cpu or -p" << std::endl;
        std::cout << "\t\t[Optional] --use_cuda_shared or -x" << std::endl;
        std::cout << "\t\t[Optional] --use_cuda_basic or -b" << std::endl;
        std::cout << "\t\t[Optional] --use_thrust or -n" << std::endl;

        exit(0);
    }

    opts->c = false;
    opts->use_cpu = false;
    opts->use_cuda_shared = false;
    opts->use_cuda_basic = false;
    opts->use_thrust = false;

    struct option l_opts[] = {
        {"num_cluster", required_argument, NULL, 'k'},
        {"dims", required_argument, NULL, 'd'},
        {"inputfilename", required_argument, NULL, 'i'},
        {"max_num_iter", required_argument, NULL, 'm'},
        {"threshold", required_argument, NULL, 't'},
        {"seed", required_argument, NULL, 's'},
        {"output_control", optional_argument, NULL, 'c'},
        {"use_cpu", optional_argument, NULL, 'p'},
        {"use_cuda_shared", optional_argument, NULL, 'x'},
        {"use_cuda_basic", optional_argument, NULL, 'b'},
        {"use_thrust", optional_argument, NULL, 'n'}
    };

    int ind, c;
    while ((c = getopt_long(argc, argv, "k:d:i:m:t:s:cpxbn", l_opts, &ind)) != -1)
    {
        switch (c)
        {
        case 0:
            break;
        case 'k':
            opts->num_cluster = atoi(optarg);
            break;
        case 'd':
            opts->dims = atoi(optarg);
            break;
        case 'i':
            opts->inputfilename = (char *)optarg;
            break;
        case 'm':
            opts->max_num_iter = atoi(optarg);
            break;
        case 't':
            opts->threshold = atof(optarg);
            break;
        case 's':
            opts->seed = atoi(optarg);
            break;
        case 'c':
            opts->c = true;
            break;
        case 'p':
            opts->use_cpu = true;
            break;
        case 'x':
            opts->use_cuda_shared = true;
            break;
        case 'b':
            opts->use_cuda_basic = true;
            break;
        case 'n':
            opts->use_thrust = true;
            break;
        case ':':
            std::cerr << argv[0] << ": option -" << (char)optopt << "requires an argument." << std::endl;
            exit(1);
        }
    }
}

void print_opts(struct options_t *opts) {
    printf("num_clusters %d\n", opts->num_cluster);
    printf("dims %d\n", opts->dims);
    printf("inputfilename %s\n", opts->inputfilename);
    printf("max_num_iter %d\n", opts->max_num_iter);
    printf("threshold %f\n", opts->threshold);
    printf("seed %d\n", opts->seed);
    if (opts->c) printf("control_flag\n");
    if (opts->use_cpu) printf("use_cpu\n");
    if (opts->use_cuda_shared) printf("use_cuda_shared\n");
    if (opts->use_cuda_basic) printf("use_cuda_basic\n");
    if (opts->use_thrust) printf("use_thrust\n");
}

