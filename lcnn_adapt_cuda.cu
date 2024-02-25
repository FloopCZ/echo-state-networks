#include "lcnn_adapt.hpp"

#include <af/cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void lcnn_adapt_kernel(
  double* prev_state,
  double* state,
  double* reservoir_w,
  double* output,
  int N,
  int M,
  int K,
  int L,
  double learning_rate,
  double weight_leakage)
{
    int block_size = blockDim.x * blockDim.y;
    int kernel_radius_height = K / 2;
    int kernel_radius_width = L / 2;

    // printf(
    //   "blockDim.x,y = %d,%d, blockIdx.x,y = %d,%d, threadIdx.x,y = %d,%d\n", blockDim.x,
    //   blockDim.y, blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);

    // TODO wrap in two grid for loops

    // Allocate the perimeter (only what will really be used).
    int perimeter_height = min(blockDim.x + K - 1, N - blockIdx.x * blockDim.x + K - 1);
    int perimeter_width = min(blockDim.y + L - 1, M - blockIdx.y * blockDim.y + L - 1);
    int perimeter_size = perimeter_height * perimeter_width;

    // Load the input matrix block with convolution perimeter to shared memory.
    // This sequentially uses all the available threads without regard to block size.
    extern __shared__ double perimeter[];
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
        perimeter[perimeter_idx] = prev_state[input_i + N * input_j];
    }

    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= N || j >= M) return;

    double postsynaptic_act = state[i + N * j];
    for (int l = 0; l < L; ++l) {
        int perimeter_j = threadIdx.y + l;
        for (int k = 0; k < K; ++k) {
            int perimeter_i = threadIdx.x + k;
            int reservoir_idx = i + N * j + N * M * k + N * M * K * l;
            int perimeter_idx = perimeter_i + perimeter_height * perimeter_j;
            double presynaptic_act = perimeter[perimeter_idx];
            double saturation_protection = pow(1 - abs(postsynaptic_act), 2);
            double out = reservoir_w[reservoir_idx] * (1. - weight_leakage)
              + saturation_protection * presynaptic_act * postsynaptic_act * learning_rate;
            out = max(0., min(1., out));
            output[reservoir_idx] = out;
        }
    }

    // printf("output i,j = %d,%d, value = %f\n", i, j, sum);
}

namespace esn {

af::array lcnn_adapt(
  const af::array& prev_state,
  const af::array& state,
  const af::array& reservoir_w,
  const lcnn_adaptation_config& cfg)
{
    // TODO add 32 bit variant
    if (prev_state.type() != af::dtype::f64 || state.type() != af::dtype::f64)
        throw std::invalid_argument(
          "CUDA kernel for lcnn adapt is only supported for 64bit arrays.");

    // Evaluate input matrices.
    prev_state.eval();
    double* pprev_state = prev_state.device<double>();
    state.eval();
    double* pstate = state.device<double>();
    reservoir_w.eval();
    double* preservoir_w = reservoir_w.device<double>();

    // Allocate output matrix.
    af::array output{reservoir_w.dims(), reservoir_w.type()};
    double* poutput = output.device<double>();

    // Determine ArrayFire's CUDA stream.
    int af_id = af::getDevice();
    cudaStream_t af_cuda_stream = afcu::getStream(af_id);

    // Call CUDA kernel.
    dim3 block(32, 32);
    dim3 grid((state.dims(0) + block.x - 1) / block.x, (state.dims(1) + block.y - 1) / block.y);
    int perimeter_bytes =
      sizeof(double) * (block.x + reservoir_w.dims(2) - 1) * (block.y + reservoir_w.dims(3) - 1);
    lcnn_adapt_kernel<<<grid, block, perimeter_bytes, af_cuda_stream>>>(
      pprev_state, pstate, preservoir_w, poutput, state.dims(0), state.dims(1), reservoir_w.dims(2),
      reservoir_w.dims(3), cfg.learning_rate, cfg.weight_leakage);
    if (cudaError_t err = cudaPeekAtLastError(); err != cudaSuccess)
        throw std::runtime_error{
          "CUDA Runtime Error in LCNN step kernel launch: " + std::string{cudaGetErrorString(err)}};

    // Give matrices back to ArrayFire.
    prev_state.unlock();
    state.unlock();
    reservoir_w.unlock();
    output.unlock();

    return output;
}

}  // namespace esn