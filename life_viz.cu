#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cstring>
#include <cuda_runtime.h>
#include "raylib.h"

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

#define TILE 16

// Window is 1280 x 720 and every cell is drawn as a 2 x 2 square
#define SCALE 2
#define WIDTH (1280 / SCALE)
#define HEIGHT (720 / SCALE)

// Same kernel as gpu_life_bench.cu
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

// Same kernel as gpu_life_tiled.cu
__global__ void step_gpu_tiled(const int* in, int* out, int width, int height)
{
    __shared__ int tile[TILE + 2][TILE + 2];

    int x = blockIdx.x * TILE + threadIdx.x;   // column
    int y = blockIdx.y * TILE + threadIdx.y;   // row

    // Board position of the top left corner of the tile, including the halo
    int start_x = blockIdx.x * TILE - 1;
    int start_y = blockIdx.y * TILE - 1;

    // Load the tile + halo, some threads load 2 cells
    for (int i = threadIdx.y * TILE + threadIdx.x; i < (TILE + 2) * (TILE + 2); i += TILE * TILE)
    {
        int tx = i % (TILE + 2);
        int ty = i / (TILE + 2);

        int gx = (start_x + tx + width) % width;
        int gy = (start_y + ty + height) % height;

        tile[ty][tx] = in[gy * width + gx];
    }

    // Wait until the whole tile is loaded before anyone reads from it
    __syncthreads();

    if (x >= width || y >= height) return; // bounds check, after the barrier

    int lx = threadIdx.x + 1;
    int ly = threadIdx.y + 1;

    // Count how many of the 8 neighbors are alive
    int neighbors = 0;

    for (int dy = -1; dy <= 1; dy++)
    {
        for (int dx = -1; dx <= 1; dx++)
        {
            if (dx == 0 && dy == 0)
            {
                continue;
            }

            neighbors += tile[ly + dy][lx + dx];
        }
    }

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

// Blend between two colors, t = 0 gives a and t = 1 gives b
__device__ uchar4 mix(uchar4 a, uchar4 b, float t)
{
    return make_uchar4(a.x + (b.x - a.x) * t,
                       a.y + (b.y - a.y) * t,
                       a.z + (b.z - a.z) * t,
                       255);
}

// Turn the grid into pixels, cells that died recently leave a fading trail
//
// heat is 1 while a cell is alive and shrinks a little every generation after it dies,
// then gets mapped onto a color gradient: white -> amber -> magenta -> indigo -> background
__global__ void colorize(const int* grid, float* heat, uchar4* pixels, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return; // bounds check

    int i = y * width + x;

    float h = grid[i] ? 1.0f : heat[i] * 0.93f;
    heat[i] = h;

    uchar4 background = make_uchar4(8, 8, 16, 255);
    uchar4 indigo     = make_uchar4(45, 15, 90, 255);
    uchar4 magenta    = make_uchar4(200, 40, 120, 255);
    uchar4 amber      = make_uchar4(255, 170, 40, 255);
    uchar4 white      = make_uchar4(255, 245, 210, 255);

    uchar4 color;

    if (h >= 1.0f)       color = white;
    else if (h > 0.6f)   color = mix(magenta, amber, (h - 0.6f) / 0.4f);
    else if (h > 0.25f)  color = mix(indigo, magenta, (h - 0.25f) / 0.35f);
    else                 color = mix(background, indigo, h / 0.25f);

    pixels[i] = color;
}

// Fill the board randomly, about 1 in 4 cells alive
void randomize(std::vector<int>& grid)
{
    for (int i = 0; i < (int)grid.size(); i++)
    {
        grid[i] = (rand() % 4 == 0) ? 1 : 0;
    }
}

// usage: ./life_viz [--screenshot]
// --screenshot runs for a few seconds, saves screenshot.png and quits
int main(int argc, char** argv)
{
    bool screenshot = argc > 1 && strcmp(argv[1], "--screenshot") == 0;

    int cells = WIDTH * HEIGHT;
    int bytes = cells * sizeof(int);

    // CPU
    std::vector<int> grid(cells, 0);
    std::vector<uchar4> pixels(cells);

    srand(42);
    randomize(grid);

    // GPU, two grids so we don't overwrite cells we still need to read
    int* d_grid;
    int* d_next;
    float* d_heat;
    uchar4* d_pixels;

    CHECK(cudaMalloc(&d_grid, bytes));
    CHECK(cudaMalloc(&d_next, bytes));
    CHECK(cudaMalloc(&d_heat, cells * sizeof(float)));
    CHECK(cudaMalloc(&d_pixels, cells * sizeof(uchar4)));

    CHECK(cudaMemcpy(d_grid, grid.data(), bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_heat, 0, cells * sizeof(float)));

    // 16x16 threads per block, round up so the blocks cover the whole board
    dim3 threads(TILE, TILE);
    dim3 blocks((WIDTH + TILE - 1) / TILE, (HEIGHT + TILE - 1) / TILE);

    // GPU timers so the HUD can show how long the simulation takes
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    InitWindow(WIDTH * SCALE, HEIGHT * SCALE, "Conway's Game of Life - CUDA");
    SetTargetFPS(60);

    // The texture the pixels get uploaded into every frame
    Image image = GenImageColor(WIDTH, HEIGHT, BLACK);
    Texture2D texture = LoadTextureFromImage(image);
    UnloadImage(image);
    SetTextureFilter(texture, TEXTURE_FILTER_POINT); // keep the cells sharp when scaled up

    bool paused = false;
    bool use_tiled = true;
    int steps_per_frame = 1;
    long generation = 0;
    float sim_ms = 0;

    while (!WindowShouldClose())
    {
        // ---- input ----

        if (IsKeyPressed(KEY_SPACE)) paused = !paused;
        if (IsKeyPressed(KEY_K)) use_tiled = !use_tiled;
        if (IsKeyPressed(KEY_UP) && steps_per_frame < 64) steps_per_frame *= 2;
        if (IsKeyPressed(KEY_DOWN) && steps_per_frame > 1) steps_per_frame /= 2;

        if (IsKeyPressed(KEY_R) || IsKeyPressed(KEY_C))
        {
            if (IsKeyPressed(KEY_R))
            {
                randomize(grid);
            }
            else
            {
                std::fill(grid.begin(), grid.end(), 0);
            }

            CHECK(cudaMemcpy(d_grid, grid.data(), bytes, cudaMemcpyHostToDevice));
            generation = 0;
        }

        // Draw with the left mouse button, erase with the right
        if (IsMouseButtonDown(MOUSE_BUTTON_LEFT) || IsMouseButtonDown(MOUSE_BUTTON_RIGHT))
        {
            int value = IsMouseButtonDown(MOUSE_BUTTON_LEFT) ? 1 : 0;
            Vector2 mouse = GetMousePosition();
            int mx = (int)mouse.x / SCALE;
            int my = (int)mouse.y / SCALE;

            // Write a small square of cells straight into the GPU grid
            for (int dy = -2; dy <= 2; dy++)
            {
                for (int dx = -2; dx <= 2; dx++)
                {
                    int nx = (mx + dx + WIDTH) % WIDTH;
                    int ny = (my + dy + HEIGHT) % HEIGHT;

                    // Leave gaps when drawing, a solid block would just die off
                    if (value == 1 && rand() % 2 == 0) continue;

                    CHECK(cudaMemcpy(d_grid + ny * WIDTH + nx, &value, sizeof(int), cudaMemcpyHostToDevice));
                }
            }
        }

        bool single_step = paused && IsKeyPressed(KEY_RIGHT);

        // ---- simulate ----

        if (!paused || single_step)
        {
            int steps = single_step ? 1 : steps_per_frame;

            CHECK(cudaEventRecord(start));

            for (int s = 0; s < steps; s++)
            {
                if (use_tiled)
                {
                    step_gpu_tiled<<<blocks, threads>>>(d_grid, d_next, WIDTH, HEIGHT);
                }
                else
                {
                    step_gpu_kernel<<<blocks, threads>>>(d_grid, d_next, WIDTH, HEIGHT);
                }
                CHECK(cudaGetLastError());

                // Swap the pointers so next becomes the current grid
                int* temp = d_grid;
                d_grid = d_next;
                d_next = temp;
            }

            CHECK(cudaEventRecord(stop));
            CHECK(cudaEventSynchronize(stop));
            CHECK(cudaEventElapsedTime(&sim_ms, start, stop));
            sim_ms /= steps;

            generation += steps;
        }

        // ---- draw ----

        // Colors get computed on the GPU, then only the finished pixels come back
        colorize<<<blocks, threads>>>(d_grid, d_heat, d_pixels, WIDTH, HEIGHT);
        CHECK(cudaGetLastError());
        CHECK(cudaMemcpy(pixels.data(), d_pixels, cells * sizeof(uchar4), cudaMemcpyDeviceToHost));

        UpdateTexture(texture, pixels.data());

        BeginDrawing();
        ClearBackground(BLACK);

        DrawTextureEx(texture, Vector2{0, 0}, 0.0f, SCALE, WHITE);

        // HUD
        DrawRectangle(8, 8, 330, 104, Color{0, 0, 0, 170});
        DrawText(TextFormat("generation  %ld", generation), 18, 16, 20, RAYWHITE);
        DrawText(TextFormat("%s kernel  %.3f ms/gen", use_tiled ? "tiled" : "naive", sim_ms), 18, 40, 20, RAYWHITE);
        DrawText(TextFormat("%d gen/frame  %d fps%s", steps_per_frame, GetFPS(), paused ? "  PAUSED" : ""), 18, 64, 20, RAYWHITE);
        DrawText("space pause  right step  up/down speed", 18, 88, 10, GRAY);
        DrawText("k kernel  r random  c clear  mouse draw", 18, 99, 10, GRAY);

        EndDrawing();

        if (screenshot && generation >= 300)
        {
            TakeScreenshot("screenshot.png");
            break;
        }
    }

    UnloadTexture(texture);
    CloseWindow();

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    CHECK(cudaFree(d_grid));
    CHECK(cudaFree(d_next));
    CHECK(cudaFree(d_heat));
    CHECK(cudaFree(d_pixels));
}
