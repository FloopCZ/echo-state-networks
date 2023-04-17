#include "lcnn_step.hpp"

#include <af/cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void
lcnn_step_kernel(double* input, double* reservoir_w, double* output, int N, int M, int K, int L)
{
    int perimeter_height = blockDim.x + K - 1;
    int perimeter_width = blockDim.y + L - 1;
    int perimeter_size = perimeter_height * perimeter_width;
    int block_size = blockDim.x * blockDim.y;
    int kernel_radius_height = K / 2;
    int kernel_radius_width = L / 2;

    extern __shared__ double perimeter[];
    // TODO wrap in two grid for loops
    // TODO add unit test

    // printf(
    //   "blockDim.x,y = %d,%d, blockIdx.x,y = %d,%d, threadIdx.x,y = %d,%d\n", blockDim.x,
    //   blockDim.y, blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);

    // Load the input matrix block with convolution perimeter to shared memory.
    // This sequentially uses all the available threads without regard to block size.
    int flat_thread_idx = threadIdx.x + blockDim.y * threadIdx.y;
    for (int perimeter_idx = flat_thread_idx; perimeter_idx < perimeter_size;
         perimeter_idx += block_size) {
        int perimeter_i = perimeter_idx % perimeter_height;
        int perimeter_j = perimeter_idx / perimeter_height;
        int input_i = blockIdx.x * blockDim.x - kernel_radius_height + perimeter_i;
        int input_j = blockIdx.y * blockDim.y - kernel_radius_width + perimeter_j;
        // Wrap around edges with correction to avoid negative modulo result.
        input_i = (input_i % N + N) % N;
        input_j = (input_j % M + M) % M;
        // printf(
        //   "perimeter_idx = %d, perimeter i,j = %d,%d, input i,j = %d,%d, value = %f\n",
        //   perimeter_idx, perimeter_i, perimeter_j, input_i, input_j, input[input_i + N *
        //   input_j]);
        perimeter[perimeter_idx] = input[input_i + N * input_j];
    }

    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= N || j >= M) return;

    double sum = 0.0;
    for (int l = 0; l < L; ++l) {
        int perimeter_j = threadIdx.y + l;
        for (int k = 0; k < K; ++k) {
            int perimeter_i = threadIdx.x + k;
            int reservoir_idx = i + N * j + N * M * k + N * M * K * l;
            int perimeter_idx = perimeter_i + perimeter_height * perimeter_j;
            // printf(
            //   "reservoir i,j,k,l = %d,%d,%d,%d, perimeter i,j = %d,%d, value = %f\n", i, j, k, l,
            //   perimeter_i, perimeter_j, perimeter[perimeter_idx]);
            sum += reservoir_w[reservoir_idx] * perimeter[perimeter_idx];
        }
    }

    // printf("output i,j = %d,%d, value = %f\n", i, j, sum);
    output[i + N * j] = sum;
}

af::array lcnn_step(const af::array& state, const af::array& reservoir_w)
{
    // TODO add 32 bit variant
    if (state.type() != af::dtype::f64)
        throw std::invalid_argument(
          "CUDA kernel for lcnn step is only supported for 64bit arrays.");

    // Evaluate input matrices.
    state.eval();
    double* pstate = state.device<double>();
    reservoir_w.eval();
    double* preservoir_w = reservoir_w.device<double>();

    // Allocate output matrix.
    af::array output{state.dims(), state.type()};
    double* poutput = output.device<double>();

    // Determine ArrayFire's CUDA stream.
    int af_id = af::getDevice();
    int cuda_id = afcu::getNativeId(af_id);
    cudaStream_t af_cuda_stream = afcu::getStream(cuda_id);

    // Call CUDA kernel.
    dim3 block(32, 32);
    dim3 grid((state.dims(0) + block.x - 1) / block.x, (state.dims(1) + block.y - 1) / block.y);
    int perimeter_bytes =
      sizeof(double) * (block.x + reservoir_w.dims(2) - 1) * (block.y + reservoir_w.dims(3) - 1);
    lcnn_step_kernel<<<grid, block, perimeter_bytes, af_cuda_stream>>>(
      pstate, preservoir_w, poutput, state.dims(0), state.dims(1), reservoir_w.dims(2),
      reservoir_w.dims(3));
    cudaStreamSynchronize(af_cuda_stream);

    // Give matrices back to ArrayFire.
    state.unlock();
    reservoir_w.unlock();
    output.unlock();

    return output;
}