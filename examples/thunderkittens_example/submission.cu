// Adapted from
// https://github.com/HazyResearch/ThunderKittens/blob/tk_gen/simple_kernels/micro_add/micro.cu
// Test whether TK works on Modal runners.

#include "task.h"
#include "utils.h"

#include "kittens.cuh"
using namespace kittens;

#define CudaCheckError() __cudaCheckError(__FILE__, __LINE__)
inline void __cudaCheckError(const char *file, const int line) {
  cudaError err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n", file, line,
            cudaGetErrorString(err));
    exit(-1);
  }
  err = cudaDeviceSynchronize();
  if (cudaSuccess != err) {
    fprintf(stderr, "cudaCheckError() with sync failed at %s:%i : %s\n", file,
            line, cudaGetErrorString(err));
    exit(-1);
  }
}

#define NUM_THREADS (kittens::WARP_THREADS) // use 1 warp

#define _row 16
#define _col 32

struct micro_globals {
  using _gl = gl<float, -1, -1, -1, -1, st_fl<_row, _col>>;
  _gl x, o;
};

__global__
__launch_bounds__(NUM_THREADS,
                  1) void micro_tk(const __grid_constant__ micro_globals g) {

  // shared memory
  extern __shared__ alignment_dummy __shm[];
  shared_allocator al((int *)&__shm[0]);
  st_fl<_row, _col>(&x_s) = al.allocate<st_fl<_row, _col>>();
  st_fl<_row, _col>(&o_s) = al.allocate<st_fl<_row, _col>>();

  // register memory
  rt_fl<_row, _col> x_reg_fl;

  // load from HBM to shared
  load(x_s, g.x, {0, 0, 0, 0});
  __syncthreads();

  // load from shared to register
  load(x_reg_fl, x_s);
  __syncthreads();

  // x (dst) = x (src b) + x (src a)
  add(x_reg_fl, x_reg_fl, x_reg_fl);
  __syncthreads();

  // store from register to shared
  store(o_s, x_reg_fl);
  __syncthreads();

  // store from shared to HBM
  store(g.o, o_s, {0, 0, 0, 0});
  __syncthreads();
}

void dispatch_micro(float *d_x, float *d_o, int N) {
  using _gl = gl<float, -1, -1, -1, -1, st_fl<_row, _col>>;
  using globals = micro_globals;
  _gl x_arg{d_x, 1, 1, _row, _col};
  _gl o_arg{d_o, 1, 1, _row, _col};
  globals g{x_arg, o_arg};
  unsigned long mem_size = 50480;
  cudaFuncSetAttribute(micro_tk, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       mem_size);

  micro_tk<<<1, 32, mem_size>>>(g);

  cudaError_t err;
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
  }
  cudaDeviceSynchronize();
}

output_t custom_kernel(input_t data) {
  output_t result;
  cudaError_t err;

  for (int i = 0; i < N_SIZES; ++i) {
    int N = Ns[i];
    result[i].resize(N);

    // Allocate device memory
    float *d_input, *d_output;
    err = cudaMalloc(&d_input, N * sizeof(float));
    if (err != cudaSuccess) {
      printf("CUDA malloc failed for d_input: %s\n", cudaGetErrorString(err));
      return result;
    }
    err = cudaMalloc(&d_output, N * sizeof(float));
    if (err != cudaSuccess) {
      printf("CUDA malloc failed for d_output: %s\n", cudaGetErrorString(err));
      return result;
    }

    // Copy input to device
    err = cudaMemcpy(d_input, data[i].data(), N * sizeof(float),
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      printf("CUDA memcpy HostToDevice failed: %s\n", cudaGetErrorString(err));
      return result;
    }

    // for (int j = 0; j < N; j++) {
    //   std::cout << data[i][j] << std::endl;
    // }
    // Copy input to device

    cudaDeviceSynchronize();
    CudaCheckError();
    dispatch_micro(d_input, d_output, N);
    cudaDeviceSynchronize();
    CudaCheckError();

    // Copy result back to host
    err = cudaMemcpy(result[i].data(), d_output, N * sizeof(float),
                     cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
      printf("CUDA memcpy DeviceToHost failed: %s\n", cudaGetErrorString(err));
      return result;
    }

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
  }

  return result;
}
