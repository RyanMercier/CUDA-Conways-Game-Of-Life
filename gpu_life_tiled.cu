#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h> // named ranges that show up in the Nsight Systems timeline

// Stop the program with the line number if a CUDA call fails
#define CHECK(call)                                                                  \
    {                                                                                \
        cudaError_t err = call;                                                      \
        if (err != cudaSuccess)                                                      \
        {                                                                            \
            printf("CUDA error %s at line %d\n", cudaGetErrorString(err), __LINE__); \
            exit(1);                                                                 \
        }                                                                            \
    }

// Each block handles a TILE x TILE square of cells
#define TILE 16

// Run one generation on the GPU using shared memory
//
// In the naive version every cell gets read from global memory 9 times (once by itself
// and once by each of its 8 neighbors). Here each block copies its tile into shared memory
// once, plus a 1 cell border (the halo) so the cells on the edge of the tile can see their
// neighbors, then all the neighbor counting reads from the fast shared copy.
//
//   h h h h h h
//   h c c c c h     c = cells this block computes (TILE x TILE)
//   h c c c c h     h = halo, neighbors that belong to other blocks
//   h c c c c h
//   h c c c c h
//   h h h h h h
__global__ void step_gpu_tiled(const int* in, int* out, int width, int height)
{
    __shared__ int tile[TILE + 2][TILE + 2];

    int x = blockIdx.x * TILE + threadIdx.x;   // column
    int y = blockIdx.y * TILE + threadIdx.y;   // row

    // Board position of the top left corner of the tile, including the halo
    int start_x = blockIdx.x * TILE - 1;
    int start_y = blockIdx.y * TILE - 1;

    // The tile + halo is 18 x 18 = 324 cells but there are only 16 x 16 = 256 threads,
    // so walk through it in steps of 256 and some threads load 2 cells
    for (int i = threadIdx.y * TILE + threadIdx.x; i < (TILE + 2) * (TILE + 2); i += TILE * TILE)
    {
        int tx = i % (TILE + 2);
        int ty = i / (TILE + 2);

        // Wrap around the edges, same as the naive version
        int gx = (start_x + tx + width) % width;
        int gy = (start_y + ty + height) % height;

        tile[ty][tx] = in[gy * width + gx];
    }

    // Wait until the whole tile is loaded before anyone reads from it
    __syncthreads();

    // Bounds check has to come after __syncthreads, if some threads returned early
    // the rest of the block would wait at the barrier for them
    if (x >= width || y >= height) return;

    // This thread's cell inside the tile, +1 to skip over the halo
    int lx = threadIdx.x + 1;
    int ly = threadIdx.y + 1;

    // Count how many of the 8 neighbors are alive
    int neighbors = 0;

    for (int dy = -1; dy <= 1; dy++)
    {
        for (int dx = -1; dx <= 1; dx++)
        {
            // Don't count the cell itself
            if (dx == 0 && dy == 0)
            {
                continue;
            }

            // No wrapping needed here, the halo already holds the wrapped neighbors
            neighbors += tile[ly + dy][lx + dx];
        }
    }

    // Rules: alive with 2 or 3 neighbors stays alive, dead with 3 comes alive, everything else dies
    int alive = tile[ly][lx];

    if (neighbors == 3 || (alive && neighbors == 2))
    {
        out[y * width + x] = 1;
    }
    else
    {
        out[y * width + x] = 0;
    }
}

// usage: ./gpu_life_tiled [width] [height] [generations]
int main(int argc, char** argv)
{
    int width = 4096;
    int height = 4096;
    int generations = 1000;

    if (argc > 1) width = atoi(argv[1]);
    if (argc > 2) height = atoi(argv[2]);
    if (argc > 3) generations = atoi(argv[3]);

    // cudaMalloc and cudaMemcpy count bytes, the vector counts ints
    int cells = width * height;
    int bytes = cells * sizeof(int);

    // CPU
    std::vector<int> grid(cells, 0);

    // Random starting board, same seed as the other versions so the results can be compared
    srand(42);
    for (int i = 0; i < cells; i++)
    {
        grid[i] = rand() % 2;
    }

    // GPU, two grids so we don't overwrite cells we still need to read
    int* d_grid;
    int* d_next;

    CHECK(cudaMalloc(&d_grid, bytes));
    CHECK(cudaMalloc(&d_next, bytes));

    // Send the starting grid to the GPU once, it stays there for every generation
    CHECK(cudaMemcpy(d_grid, grid.data(), bytes, cudaMemcpyHostToDevice));

    // The block has to be exactly TILE x TILE, the kernel's shared memory is sized for it
    dim3 threads(TILE, TILE);
    dim3 blocks((width + TILE - 1) / TILE, (height + TILE - 1) / TILE);

    // Warm up launch so the first real generation doesn't pay for startup costs
    // It writes into d_next which the first real generation overwrites anyway
    step_gpu_tiled<<<blocks, threads>>>(d_grid, d_next, width, height);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    // GPU timers, they measure time on the GPU itself
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    nvtxRangePushA("tiled simulate");
    CHECK(cudaEventRecord(start));

    for (int generation = 0; generation < generations; generation++)
    {
        step_gpu_tiled<<<blocks, threads>>>(d_grid, d_next, width, height);

        // Launches don't return an error, this catches bad launch settings
        CHECK(cudaGetLastError());

        // Swap the pointers so next becomes the current grid
        int* temp = d_grid;
        d_grid = d_next;
        d_next = temp;
    }

    CHECK(cudaEventRecord(stop));

    // Launches return right away, wait for the GPU to actually finish
    // This also catches errors that happen while the kernel is running
    CHECK(cudaEventSynchronize(stop));
    nvtxRangePop();

    float ms = 0;
    CHECK(cudaEventElapsedTime(&ms, start, stop));

    // Bring the final grid back once at the end
    CHECK(cudaMemcpy(grid.data(), d_grid, bytes, cudaMemcpyDeviceToHost));

    // Count live cells at the end, if this matches the other versions they computed the same thing
    long live = 0;
    for (int i = 0; i < cells; i++)
    {
        live += grid[i];
    }

    double cells_per_second = (double)cells * generations / (ms / 1000.0);

    printf("tiled gpu  | %d x %d | %d generations\n", width, height, generations);
    printf("time:        %.2f ms (%.3f ms per generation)\n", ms, ms / generations);
    printf("speed:       %.3e cells/s\n", cells_per_second);
    printf("live cells:  %ld\n", live);

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    CHECK(cudaFree(d_grid));
    CHECK(cudaFree(d_next));
}
