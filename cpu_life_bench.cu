#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>

// Run one generation on the CPU, reads from in and writes to out
void step_cpu(const std::vector<int>& in, std::vector<int>& out, int width, int height)
{
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
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
    }
}

// usage: ./cpu_life_bench [width] [height] [generations]
int main(int argc, char** argv)
{
    int width = 4096;
    int height = 4096;
    int generations = 10; // the CPU is slow, keep this small

    if (argc > 1) width = atoi(argv[1]);
    if (argc > 2) height = atoi(argv[2]);
    if (argc > 3) generations = atoi(argv[3]);

    // Two grids so we don't overwrite cells we still need to read
    std::vector<int> grid(width * height, 0);
    std::vector<int> next(width * height, 0);

    // Random starting board, same seed as the GPU versions so the results can be compared
    srand(42);
    for (int i = 0; i < width * height; i++)
    {
        grid[i] = rand() % 2;
    }

    auto start = std::chrono::high_resolution_clock::now();

    for (int generation = 0; generation < generations; generation++)
    {
        step_cpu(grid, next, width, height);

        // Swap so next becomes the current grid
        grid.swap(next);
    }

    auto stop = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(stop - start).count();

    // Count live cells at the end, if this matches the GPU versions they computed the same thing
    long live = 0;
    for (int i = 0; i < width * height; i++)
    {
        live += grid[i];
    }

    double cells_per_second = (double)width * height * generations / (ms / 1000.0);

    printf("cpu        | %d x %d | %d generations\n", width, height, generations);
    printf("time:        %.2f ms (%.3f ms per generation)\n", ms, ms / generations);
    printf("speed:       %.3e cells/s\n", cells_per_second);
    printf("live cells:  %ld\n", live);
}
