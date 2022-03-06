CC = nvcc
CFLAGS = --std=c++11 -Werror cross-execution-space-call -lm
SOURSES = main.cu
BIN = main
all:
	$(CC) $(CFLAGS) -o $(BIN) $(SOURSES)
