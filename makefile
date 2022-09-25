CC = g++
SRCS = ./src/*.cpp
INC = ./src/
OPTS = -std=c++17 -Wall -Werror

EXEC = bin/kmeans

all: clean compile

compile:
	$(CC) $(SRCS) $(OPTS) -O3 -I$(INC) -o $(EXEC)

debug:
	$(CC) $(SRCS) $(OPTS) -g -I$(INC) -o $(EXEC)

clean:
	rm -f $(EXEC)
