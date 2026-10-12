#include <cstdio>
#include <vector>

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

int main()
{
    int width = 16;
    int height = 16;
    int generations = 8;

    // Two grids so we don't overwrite cells we still need to read
    std::vector<int> grid(width * height, 0);
    std::vector<int> next(width * height, 0);

    // Glider in the top left corner
    // .#.
    // ..#
    // ###
    grid[0 * width + 1] = 1;
    grid[1 * width + 2] = 1;
    grid[2 * width + 0] = 1;
    grid[2 * width + 1] = 1;
    grid[2 * width + 2] = 1;

    for (int generation = 0; generation <= generations; generation++)
    {
        printf("Generation %d\n", generation);
        print_grid(grid, width, height);

        step_cpu(grid, next, width, height);

        // Swap so next becomes the current grid
        grid.swap(next);
    }
}
