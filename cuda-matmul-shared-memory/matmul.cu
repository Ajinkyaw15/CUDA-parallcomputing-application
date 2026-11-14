// Matrix multiplication in CUDA: global-memory kernel vs shared-memory (tiled) kernel.
// Build: nvcc -O3 -o matmul matmul.cu   (optionally -DTILE=32)
// Run:   ./matmul [N1 N2 ...]            (default sizes: 256 512 1024 2048)

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#ifndef TILE
#define TILE 16
#endif

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error: %s at %s:%d\n",                       \
                    cudaGetErrorString(err__), __FILE__, __LINE__);            \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// Kernel 1: naive. Every thread reads a full row of A and column of B straight
// from global memory: 2*N global loads per output element.
__global__ void matmul_global(const float *A, const float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k)
            sum += A[row * N + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

// Kernel 2: tiled. Each block cooperatively loads a TILE x TILE tile of A and
// B into shared memory, so each global element is loaded once per block and
// reused TILE times from fast on-chip memory.
__global__ void matmul_shared(const float *A, const float *B, float *C, int N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    float sum = 0.0f;
    int numTiles = (N + TILE - 1) / TILE;

    for (int t = 0; t < numTiles; ++t) {
        int aCol = t * TILE + tx;
        int bRow = t * TILE + ty;
        As[ty][tx] = (row < N && aCol < N) ? A[row * N + aCol] : 0.0f;
        Bs[ty][tx] = (bRow < N && col < N) ? B[bRow * N + col] : 0.0f;
        __syncthreads();                       // tile fully loaded

#pragma unroll
        for (int k = 0; k < TILE; ++k)
            sum += As[ty][k] * Bs[k][tx];
        __syncthreads();                       // done before next overwrite
    }
    if (row < N && col < N)
        C[row * N + col] = sum;
}

// CPU reference (used for small N only).
static void matmul_cpu(const float *A, const float *B, float *C, int N) {
    for (int i = 0; i < N; ++i)
        for (int k = 0; k < N; ++k) {
            float a = A[i * N + k];
            for (int j = 0; j < N; ++j)
                C[i * N + j] += a * B[k * N + j];
        }
}

static float max_abs_diff(const float *a, const float *b, size_t n) {
    float m = 0.0f;
    for (size_t i = 0; i < n; ++i) m = fmaxf(m, fabsf(a[i] - b[i]));
    return m;
}

typedef void (*Kernel)(const float *, const float *, float *, int);

// Average kernel time in ms over `reps` runs after one warm-up launch.
static float time_kernel(Kernel k, const float *dA, const float *dB, float *dC,
                         int N, int reps) {
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);
    k<<<grid, block>>>(dA, dB, dC, N);          // warm-up
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t s, e;
    CUDA_CHECK(cudaEventCreate(&s));
    CUDA_CHECK(cudaEventCreate(&e));
    CUDA_CHECK(cudaEventRecord(s));
    for (int i = 0; i < reps; ++i) k<<<grid, block>>>(dA, dB, dC, N);
    CUDA_CHECK(cudaEventRecord(e));
    CUDA_CHECK(cudaEventSynchronize(e));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));
    CUDA_CHECK(cudaEventDestroy(s));
    CUDA_CHECK(cudaEventDestroy(e));
    return ms / reps;
}

int main(int argc, char **argv) {
    std::vector<int> sizes;
    for (int i = 1; i < argc; ++i) sizes.push_back(atoi(argv[i]));
    if (sizes.empty()) sizes = {256, 512, 1024, 2048};

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s | TILE = %d\n\n", prop.name, TILE);
    printf("%6s %12s %12s %10s %10s %10s %12s\n", "N", "global(ms)", "shared(ms)",
           "speedup", "GFLOPS_g", "GFLOPS_s", "max_err");

    FILE *csv = fopen("results.csv", "w");
    if (csv) fprintf(csv, "N,tile,global_ms,shared_ms,speedup,gflops_global,gflops_shared,max_err\n");

    for (int N : sizes) {
        size_t n = (size_t)N * N, bytes = n * sizeof(float);
        std::vector<float> hA(n), hB(n), hCg(n), hCs(n), hRef(n, 0.0f);
        srand(42);
        for (size_t i = 0; i < n; ++i) {
            hA[i] = rand() / (float)RAND_MAX;
            hB[i] = rand() / (float)RAND_MAX;
        }

        float *dA, *dB, *dC;
        CUDA_CHECK(cudaMalloc(&dA, bytes));
        CUDA_CHECK(cudaMalloc(&dB, bytes));
        CUDA_CHECK(cudaMalloc(&dC, bytes));
        CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice));

        int reps = (N <= 512) ? 50 : 10;
        float tg = time_kernel(matmul_global, dA, dB, dC, N, reps);
        CUDA_CHECK(cudaMemcpy(hCg.data(), dC, bytes, cudaMemcpyDeviceToHost));
        float ts = time_kernel(matmul_shared, dA, dB, dC, N, reps);
        CUDA_CHECK(cudaMemcpy(hCs.data(), dC, bytes, cudaMemcpyDeviceToHost));

        // Correctness: vs CPU for N <= 512, otherwise global vs shared.
        float err;
        if (N <= 512) {
            matmul_cpu(hA.data(), hB.data(), hRef.data(), N);
            err = fmaxf(max_abs_diff(hRef.data(), hCg.data(), n),
                        max_abs_diff(hRef.data(), hCs.data(), n));
        } else {
            err = max_abs_diff(hCg.data(), hCs.data(), n);
        }

        double flops = 2.0 * N * N * (double)N;
        double gg = flops / (tg * 1e6), gs = flops / (ts * 1e6);
        printf("%6d %12.3f %12.3f %9.2fx %10.1f %10.1f %12.2e\n", N, tg, ts,
               tg / ts, gg, gs, err);
        if (csv) fprintf(csv, "%d,%d,%.4f,%.4f,%.3f,%.2f,%.2f,%.3e\n", N, TILE, tg, ts, tg / ts, gg, gs, err);

        cudaFree(dA); cudaFree(dB); cudaFree(dC);
    }
    if (csv) fclose(csv);
    return 0;
}
