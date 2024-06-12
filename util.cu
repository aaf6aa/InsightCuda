#pragma once

#include <stdexcept>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>

// suppress kernel launch intellisense bug
#ifdef __INTELLISENSE__
#define KERNEL_ARGS2(grid_size, block_size)
#define KERNEL_ARGS3(grid_size, block_size, shared_mem)
#define KERNEL_ARGS4(grid_size, block_size, shared_mem, stream)
#else
#define KERNEL_ARGS2(grid_size, block_size) <<< grid_size, block_size >>>
#define KERNEL_ARGS3(grid_size, block_size, shared_mem) <<< grid_size, block_size, shared_mem >>>
#define KERNEL_ARGS4(grid_size, block_size, shared_mem, stream) <<< grid_size, block_size, shared_mem, stream >>>
#endif

#define CUDA_ERROR_CHECK(cuda_status) if (cuda_status != cudaSuccess) throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorName(cuda_status) + ": " + cudaGetErrorString(cuda_status) + " at " + __FILE__ + ":" + std::to_string(__LINE__))
