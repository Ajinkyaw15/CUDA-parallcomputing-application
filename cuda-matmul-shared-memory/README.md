# CUDA Matrix Multiplication: Global vs Shared Memory

Compares two CUDA kernels computing `C = A x B` for square `N x N` float matrices:

| Kernel | Idea |
|---|---|
| `matmul_global` | Each thread computes one `C[i][j]`, reading its row of A and column of B directly from global memory. |
| `matmul_shared` | Each block loads `TILE x TILE` tiles of A and B into shared memory, synchronizes, accumulates, and moves to the next tile. |

The program checks correctness (vs a CPU reference for N <= 512, otherwise global vs shared), times both kernels with CUDA events, and writes `results.csv`.

## Build and run

```bash
make                 # needs nvcc; ARCH=native by default (e.g. make ARCH=sm_75)
./matmul             # sizes 256 512 1024 2048
./matmul 4096 8192   # custom sizes
make TILE=32         # try a different tile size
```

## Why shared memory helps

- **Global kernel:** every output element does `2N` global loads. Neighbouring threads re-read the same rows/columns, so total global traffic is about `2N^3` floats. The kernel is memory-bound.
- **Tiled kernel:** each element of A and B is loaded from global memory once per tile and reused `TILE` times from shared memory. Global traffic drops by a factor of about `TILE` (about `2N^3 / TILE`), and arithmetic intensity rises accordingly.
- **Access patterns:** in both kernels, loads from global memory are coalesced across `threadIdx.x`. In the tiled kernel, `Bs[k][tx]` is conflict-free and `As[ty][k]` is a broadcast, so no padding is needed.
- **Cost:** two `__syncthreads()` per tile and shared-memory usage of `2 * TILE^2 * 4` bytes per block (2 KB at TILE=16).

## Expected behaviour

- Small N (<= 256): little difference, since launch overhead and the L2 cache dominate.
- Larger N: the tiled kernel is typically several times faster, and the gap grows with N.
- Larger TILE (32) reduces global traffic further, but uses more shared memory and registers per block and can lower occupancy.
- Neither kernel approaches cuBLAS. Register tiling, vectorized loads and tensor cores are the next steps.

## Results

Run `./matmul` on your GPU and paste the table here (results depend on the GPU):

| GPU | N | Global (ms) | Shared (ms) | Speedup |
|---|---|---|---|---|
| _fill in_ | | | | |

## Possible extensions

- Compare against cuBLAS `cublasSgemm`.
- Add 2x2 register blocking per thread.
- Profile with Nsight Compute (global load transactions, shared-memory bank conflicts, occupancy).
- Sweep TILE in {8, 16, 32}.
