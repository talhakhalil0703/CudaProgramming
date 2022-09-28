CC = g++
SRCS = ./src/*.cpp
SRCSS = ./src/*.cu
INC = ./src/
OPTS = -std=c++17 -Wall -Werror

EXEC = bin/kmeans

all: clean compile

compile:
	$(CC) $(SRCS) $(OPTS) -O3 -I$(INC) -o $(EXEC)

debug:
	$(CC) $(SRCS) $(OPTS) -g -I$(INC) -o $(EXEC)_debug

clean:
	rm -f $(EXEC)

cuda:
	nvcc $(SRCS) $(SRCSS) -O3 -I$(INC) -o $(EXEC)_cuda

cuda_debug:
	nvcc $(SRCS) $(SRCSS) -g -I$(INC) -o $(EXEC)_cuda_debug
