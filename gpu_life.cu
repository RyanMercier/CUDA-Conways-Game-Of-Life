#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

// Print the grid, # is alive and . is dead
void print_grid(const std::vector<int>& grid, int width, int height)
{
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            printf("%c", grid[y * width + x] ? '#' : '.');
        }
        printf("\n");
    }
    printf("\n");
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

int main()
{
    int width = 16;
    int height = 16;
    int generations = 8;

    // Two grids so we don't overwrite cells we still need to read
    int bytes = width * height * sizeof(int);
    
    // CPU
    std::vector<int> grid(width * height, 0);

    //GPU
    int* d_grid;
    int* d_next;

    cudaMalloc(&d_grid, bytes);
    cudaMalloc(&d_next, bytes);

    // Glider in the top left corner
    // .#.
    // ..#
    // ###
    grid[0 * width + 1] = 1;
    grid[1 * width + 2] = 1;
    grid[2 * width + 0] = 1;
    grid[2 * width + 1] = 1;
    grid[2 * width + 2] = 1;

    cudaMemcpy(d_grid, grid.data(), bytes, cudaMemcpyHostToDevice);

    // 16x16 threads per block, round up so the blocks cover the whole board
    dim3 threads(16, 16);
    dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);

    for (int generation = 0; generation <= generations; generation++)
    {
        // Bring the current grid back to the CPU so we can print it
        cudaMemcpy(grid.data(), d_grid, bytes, cudaMemcpyDeviceToHost);
        
        printf("Generation %d\n", generation);
        print_grid(grid, width, height);

        step_gpu_kernel<<<blocks, threads>>>(d_grid, d_next, width, height);

        // Swap so next becomes the current grid
        int* temp = d_grid;
        d_grid = d_next;
        d_next = temp;
    }

    cudaFree(d_grid);
    cudaFree(d_next);
}
