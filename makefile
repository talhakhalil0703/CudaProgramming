CC = g++
SRCS = ./src/*.cpp
SRCSS = ./src/*.cu
INC = ./src/
OPTS = -std=c++17 -Wall -Werror

EXEC = bin/kmeans

all: clean compile test

clean:
	rm -f $(EXEC)

compile:
	nvcc $(SRCS) $(SRCSS) -O3 -I$(INC) -o $(EXEC)

debug:
	nvcc $(SRCS) $(SRCSS) -g -I$(INC) -o $(EXEC)_debug

test:
	python3 tester.py
