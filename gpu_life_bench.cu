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

// Run one generation on the GPU, reads from in and writes to out
__global__ void step_gpu_kernel(const int* in, int* out, int width, int height)
{
    // Count how many of the 8 neighbors are alive
    int neighbors = 0;

    int x = blockIdx.x * blockDim.x + threadIdx.x;   // column
    int y = blockIdx.y * blockDim.y + threadIdx.y;   // row

    if (x >= width || y >= height) return; // bounds check

    for (int dy = -1; dy <= 1; dy++)
    {
        for (int dx = -1; dx <= 1; dx++)
        {
            // Don't count the cell itself
            if (dx == 0 && dy == 0)
            {
                continue;
            }

            // Wrap around the edges, add width/height first so it never goes negative
            int nx = (x + dx + width) % width;
            int ny = (y + dy + height) % height;
            int index = ny * width + nx;

            neighbors += in[index];
        }
    }

    // Rules: alive with 2 or 3 neighbors stays alive, dead with 3 comes alive, everything else dies
    int alive = in[y * width + x];

    if (neighbors == 3 || (alive && neighbors == 2))
    {
        out[y * width + x] = 1;
    }
    else
    {
        out[y * width + x] = 0;
    }
}

// usage: ./gpu_life_bench [width] [height] [generations]
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

    // Random starting board, same seed as the CPU version so the results can be compared
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

    // 16x16 threads per block, round up so the blocks cover the whole board
    dim3 threads(16, 16);
    dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);

    // Warm up launch so the first real generation doesn't pay for startup costs
    // It writes into d_next which the first real generation overwrites anyway
    step_gpu_kernel<<<blocks, threads>>>(d_grid, d_next, width, height);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    // GPU timers, they measure time on the GPU itself
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    nvtxRangePushA("naive simulate");
    CHECK(cudaEventRecord(start));

    for (int generation = 0; generation < generations; generation++)
    {
        step_gpu_kernel<<<blocks, threads>>>(d_grid, d_next, width, height);

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

    // Count live cells at the end, if this matches the CPU version they computed the same thing
    long live = 0;
    for (int i = 0; i < cells; i++)
    {
        live += grid[i];
    }

    double cells_per_second = (double)cells * generations / (ms / 1000.0);

    printf("naive gpu  | %d x %d | %d generations\n", width, height, generations);
    printf("time:        %.2f ms (%.3f ms per generation)\n", ms, ms / generations);
    printf("speed:       %.3e cells/s\n", cells_per_second);
    printf("live cells:  %ld\n", live);

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    CHECK(cudaFree(d_grid));
    CHECK(cudaFree(d_next));
}
