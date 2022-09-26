CC = g++
SRCS = ./src/*.cpp
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
