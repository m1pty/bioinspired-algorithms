CC = gcc
CFLAGS = -O2 -std=c11 -Wall -Wextra -fopenmp
LDFLAGS = -lm -fopenmp

INCLUDES = -Iinclude
SRC = src/main.c src/selection.c src/mutation.c src/crossover.c include/ga.h

all: main

main: $(SRC)
	$(CC) $(CFLAGS) $(INCLUDES) $(SRC) -o main $(LDFLAGS)

clean:
	rm -f main