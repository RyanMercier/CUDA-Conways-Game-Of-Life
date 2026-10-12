NVCC = nvcc
FLAGS = -O3 -arch=sm_89
RAYLIB = -L/usr/local/lib -lraylib -lGL -lm -lX11

# Ubuntu's packaged nsys/ncu (2022.4) are too old for driver 561, these are the CUDA 12.6 versions
NSYS = $(HOME)/.local/nsight/root/opt/nvidia/nsight-systems/2024.5.1/target-linux-x64/nsys
NCU = $(HOME)/.local/nsight/root/opt/nvidia/nsight-compute/2024.3.2/ncu

all: cpu_life_bench gpu_life_bench gpu_life_tiled gpu_life_bits life_viz

cpu_life_bench: cpu_life_bench.cu
	$(NVCC) $(FLAGS) -o $@ $<

gpu_life_bench: gpu_life_bench.cu
	$(NVCC) $(FLAGS) -o $@ $<

gpu_life_tiled: gpu_life_tiled.cu
	$(NVCC) $(FLAGS) -o $@ $<

gpu_life_bits: gpu_life_bits.cu
	$(NVCC) $(FLAGS) -o $@ $<

life_viz: life_viz.cu
	$(NVCC) $(FLAGS) -o $@ $< $(RAYLIB)

# All four should print the same live cell count
check: cpu_life_bench gpu_life_bench gpu_life_tiled gpu_life_bits
	./cpu_life_bench 1024 768 50
	./gpu_life_bench 1024 768 50
	./gpu_life_tiled 1024 768 50
	./gpu_life_bits 1024 768 50

bench: cpu_life_bench gpu_life_bench gpu_life_tiled gpu_life_bits
	./cpu_life_bench 4096 4096 10
	./gpu_life_bench 4096 4096 1000
	./gpu_life_tiled 4096 4096 1000
	./gpu_life_bits 4096 4096 1000
	./gpu_life_bench 16384 16384 100
	./gpu_life_tiled 16384 16384 100
	./gpu_life_bits 16384 16384 1000

# Nsight Systems: timeline of CUDA API calls, kernels, memcpys and the NVTX ranges
nsys: gpu_life_bench gpu_life_tiled gpu_life_bits
	mkdir -p profiles
	$(NSYS) profile --trace=cuda,nvtx -o profiles/naive --force-overwrite true ./gpu_life_bench 4096 4096 1000
	$(NSYS) profile --trace=cuda,nvtx -o profiles/tiled --force-overwrite true ./gpu_life_tiled 4096 4096 1000
	$(NSYS) profile --trace=cuda,nvtx -o profiles/bits --force-overwrite true ./gpu_life_bits 4096 4096 1000
	$(NSYS) stats --force-export=true --report cuda_api_sum,cuda_gpu_kern_sum,nvtx_sum profiles/naive.nsys-rep
	$(NSYS) stats --force-export=true --report cuda_api_sum,cuda_gpu_kern_sum,nvtx_sum profiles/tiled.nsys-rep
	$(NSYS) stats --force-export=true --report cuda_api_sum,cuda_gpu_kern_sum,nvtx_sum profiles/bits.nsys-rep

# Nsight Compute: what's limiting each kernel (memory, compute, cache hit rates)
# Needs GPU performance counters enabled in the Windows NVIDIA Control Panel
ncu: gpu_life_bench gpu_life_tiled gpu_life_bits
	mkdir -p profiles
	$(NCU) --set full -k step_gpu_kernel -s 2 -c 1 -o profiles/naive --force-overwrite ./gpu_life_bench 4096 4096 5
	$(NCU) --set full -k step_gpu_tiled -s 2 -c 1 -o profiles/tiled --force-overwrite ./gpu_life_tiled 4096 4096 5
	$(NCU) --set full -k step_gpu_bits -s 2 -c 1 -o profiles/bits --force-overwrite ./gpu_life_bits 4096 4096 5

clean:
	rm -f cpu_life_bench gpu_life_bench gpu_life_tiled gpu_life_bits life_viz cpu_life gpu_life life

.PHONY: all check bench nsys ncu clean
