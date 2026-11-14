NVCC ?= nvcc
ARCH ?= native
TILE ?= 16

all: matmul

matmul: matmul.cu
	$(NVCC) -O3 -arch=$(ARCH) -DTILE=$(TILE) -o matmul matmul.cu

run: matmul
	./matmul

clean:
	rm -f matmul results.csv
