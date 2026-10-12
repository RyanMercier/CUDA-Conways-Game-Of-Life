#include <cstdio>
#include <cstdlib>
#include <cstdint>
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

// Add one neighbor mask into a 3 bit counter, for all 32 cells at the same time
//
// c0, c1, c2 are the 1s, 2s and 4s bits of the neighbor count. Bit i of each one belongs
// to cell i, so this is 32 separate little adders running side by side in one uint32.
// We only care if the count is exactly 2 or 3, so once a cell reaches 4 its c2 bit just
// stays on (4 or more neighbors always means dead).
__device__ void add_neighbor(uint32_t mask, uint32_t& c0, uint32_t& c1, uint32_t& c2)
{
    uint32_t carry0 = c0 & mask;
    c0 = c0 ^ mask;

    uint32_t carry1 = c1 & carry0;
    c1 = c1 ^ carry0;

    c2 = c2 | carry1;
}

// Run one generation on the GPU with 1 bit per cell
//
// Each uint32 holds 32 cells from the same row, bit i is cell (word * 32 + i).
// Each thread handles one word, so 32 cells. Instead of looking at neighbors one cell at
// a time we shift whole words: shifting a row left by 1 lines every cell up with its left
// neighbor, so one shift gives the "left neighbor" of all 32 cells at once.
//
// words_per_row = width / 32
__global__ void step_gpu_bits(const uint32_t* in, uint32_t* out, int words_per_row, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;   // which word in the row
    int y = blockIdx.y * blockDim.y + threadIdx.y;   // row

    if (x >= words_per_row || y >= height) return; // bounds check

    // Wrap around the edges with an if instead of %, % is slow on the GPU
    int x_left = (x == 0) ? words_per_row - 1 : x - 1;
    int x_right = (x == words_per_row - 1) ? 0 : x + 1;
    int y_up = (y == 0) ? height - 1 : y - 1;
    int y_down = (y == height - 1) ? 0 : y + 1;

    int rows[3] = { y_up, y, y_down };

    uint32_t c0 = 0;
    uint32_t c1 = 0;
    uint32_t c2 = 0;
    uint32_t alive = 0;

    for (int r = 0; r < 3; r++)
    {
        uint32_t word = in[rows[r] * words_per_row + x];
        uint32_t word_left = in[rows[r] * words_per_row + x_left];
        uint32_t word_right = in[rows[r] * words_per_row + x_right];

        // Left neighbor of bit i is bit i - 1, bit 0's left neighbor is bit 31 of the word to the left
        uint32_t left = (word << 1) | (word_left >> 31);

        // Right neighbor of bit i is bit i + 1, bit 31's right neighbor is bit 0 of the word to the right
        uint32_t right = (word >> 1) | (word_right << 31);

        add_neighbor(left, c0, c1, c2);
        add_neighbor(right, c0, c1, c2);

        // The cells directly above and below are neighbors, the middle row is the cells themselves
        if (r == 1)
        {
            alive = word;
        }
        else
        {
            add_neighbor(word, c0, c1, c2);
        }
    }

    // Rules: alive next generation if count == 3, or count == 2 and alive now
    // count is 2 or 3 when c1 is on and c2 is off, then c0 picks between them (0 = 2, 1 = 3)
    out[y * words_per_row + x] = c1 & ~c2 & (c0 | alive);
}

// usage: ./gpu_life_bits [width] [height] [generations], width has to be a multiple of 32
int main(int argc, char** argv)
{
    int width = 4096;
    int height = 4096;
    int generations = 1000;

    if (argc > 1) width = atoi(argv[1]);
    if (argc > 2) height = atoi(argv[2]);
    if (argc > 3) generations = atoi(argv[3]);

    if (width % 32 != 0)
    {
        printf("width has to be a multiple of 32\n");
        return 1;
    }

    int cells = width * height;
    int words_per_row = width / 32;
    int words = words_per_row * height;
    int bytes = words * sizeof(uint32_t);

    // Random starting board, same seed as the other versions so the results can be compared
    std::vector<int> start(cells, 0);
    srand(42);
    for (int i = 0; i < cells; i++)
    {
        start[i] = rand() % 2;
    }

    // Pack 32 cells into each word
    std::vector<uint32_t> grid(words, 0);
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            if (start[y * width + x])
            {
                grid[y * words_per_row + x / 32] |= 1u << (x % 32);
            }
        }
    }

    // GPU, two grids so we don't overwrite cells we still need to read
    uint32_t* d_grid;
    uint32_t* d_next;

    CHECK(cudaMalloc(&d_grid, bytes));
    CHECK(cudaMalloc(&d_next, bytes));

    // Send the starting grid to the GPU once, it stays there for every generation
    CHECK(cudaMemcpy(d_grid, grid.data(), bytes, cudaMemcpyHostToDevice));

    // 32 x 8 threads per block, 32 wide so each warp reads one long run of a row
    dim3 threads(32, 8);
    dim3 blocks((words_per_row + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);

    // Warm up launch so the first real generation doesn't pay for startup costs
    // It writes into d_next which the first real generation overwrites anyway
    step_gpu_bits<<<blocks, threads>>>(d_grid, d_next, words_per_row, height);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    // GPU timers, they measure time on the GPU itself
    cudaEvent_t start_event, stop_event;
    CHECK(cudaEventCreate(&start_event));
    CHECK(cudaEventCreate(&stop_event));

    nvtxRangePushA("bits simulate");
    CHECK(cudaEventRecord(start_event));

    for (int generation = 0; generation < generations; generation++)
    {
        step_gpu_bits<<<blocks, threads>>>(d_grid, d_next, words_per_row, height);

        // Launches don't return an error, this catches bad launch settings
        CHECK(cudaGetLastError());

        // Swap the pointers so next becomes the current grid
        uint32_t* temp = d_grid;
        d_grid = d_next;
        d_next = temp;
    }

    CHECK(cudaEventRecord(stop_event));

    // Launches return right away, wait for the GPU to actually finish
    // This also catches errors that happen while the kernel is running
    CHECK(cudaEventSynchronize(stop_event));
    nvtxRangePop();

    float ms = 0;
    CHECK(cudaEventElapsedTime(&ms, start_event, stop_event));

    // Bring the final grid back once at the end
    CHECK(cudaMemcpy(grid.data(), d_grid, bytes, cudaMemcpyDeviceToHost));

    // Count live cells at the end, popcount counts the 1 bits in a word
    long live = 0;
    for (int i = 0; i < words; i++)
    {
        live += __builtin_popcount(grid[i]);
    }

    double cells_per_second = (double)cells * generations / (ms / 1000.0);

    printf("bits gpu   | %d x %d | %d generations\n", width, height, generations);
    printf("time:        %.2f ms (%.3f ms per generation)\n", ms, ms / generations);
    printf("speed:       %.3e cells/s\n", cells_per_second);
    printf("live cells:  %ld\n", live);

    CHECK(cudaEventDestroy(start_event));
    CHECK(cudaEventDestroy(stop_event));
    CHECK(cudaFree(d_grid));
    CHECK(cudaFree(d_next));
}
