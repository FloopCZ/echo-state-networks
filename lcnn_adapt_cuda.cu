#include "lcnn_adapt.hpp"

#include <af/cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void lcnn_adapt_kernel(
  double* prev_prev_state,
  double* prev_state,
  double* curr_state,
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
    extern __shared__ double shm[];
    double* perimeter_presynaptic_diff = shm;
    double* perimeter_postsynaptic_diff = shm + perimeter_size;
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
        // forward-postsynaptic = me now
        // backward-presynaptic = me before
        // forward-postsynaptic = me now
        // backward-postsynaptic = me before
        perimeter_presynaptic_diff[perimeter_idx] =
          prev_state[input_i + N * input_j] - prev_prev_state[input_i + N * input_j];
        perimeter_postsynaptic_diff[perimeter_idx] =
          curr_state[input_i + N * input_j] - prev_state[input_i + N * input_j];
    }

    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= N || j >= M) return;

    double postsynaptic_diff = curr_state[i + N * j] - prev_state[i + N * j];
    double presynaptic_diff = prev_state[i + N * j] - prev_prev_state[i + N * j];
    for (int l = 0; l < L; ++l) {
        int perimeter_j = threadIdx.y + l;
        for (int k = 0; k < K; ++k) {
            bool is_self = j == 0 && k == 0;
            int perimeter_i = threadIdx.x + k;
            int reservoir_idx = i + N * j + N * M * k + N * M * K * l;
            int perimeter_idx = perimeter_i + perimeter_height * perimeter_j;
            double presynaptic_diff_fwd = perimeter_presynaptic_diff[perimeter_idx];
            double postsynaptic_diff_fwd = postsynaptic_diff;
            double presynaptic_diff_bwd = presynaptic_diff;
            double postsynaptic_diff_bwd = perimeter_postsynaptic_diff[perimeter_idx];
            // double learning_strength = !is_self * pow(abs(curr_state[i + N * j]), 2);
            double learning_strength = !is_self;
            double delta_fwd = presynaptic_diff_fwd * postsynaptic_diff_fwd * learning_rate * learning_strength;
            double delta_bwd = presynaptic_diff_bwd * postsynaptic_diff_bwd * learning_rate * learning_strength;
            double out = reservoir_w[reservoir_idx] * (1. - weight_leakage) + delta_fwd - 2 * delta_bwd;
            // double delta = presynaptic_diff * postsynaptic_diff * learning_rate;
            // if (abs(delta) > 1e-2) {
            // printf("delta=%.10f\n", delta);
            // printf("output i,j,k,l = %d,%d,%d,%d wl=%.10f lr=%.10f\n", i, j, k, l,
            // weight_leakage, learning_rate);
            // }
            // out *= 1e10;
            // if (postsynaptic_diff > 0.5)
            //     printf("output i,j,k,l = %d,%d,%d,%d value = %.10f\n", i, j, k, l, out);
            // out = max(-1., min(1., out));
            // if (i == 1 && j == 8 && abs(postsynaptic_diff) > 1e-16) {
            //     // if (abs(delta) > 1e-6) {
            //     printf("delta=%.10f\n", delta);
            //     printf("output i,j,k,l = %d,%d,%d,%d value = %.10f\n", i, j, k, l, out);
            //     printf("presynatpic diff i,j = %d,%d, value = %.10f\n", i, j, presynaptic_diff);
            //     printf("postsynatpic diff i,j = %d,%d, value = %.10f\n", i, j, postsynaptic_diff);
            //     printf("postsynatpic i,j = %d,%d, value = %.10f\n", i, j, curr_state[i + N * j]);
            //     printf(
            //       "postsynatpic_prev i,j = %d,%d, value = %.10f\n", i, j, prev_state[i + N * j]);
            //     printf("\n");
            // }
            output[reservoir_idx] = out;
        }
    }

    // printf("output i,j = %d,%d, value = %f\n", i, j, sum);
}

namespace esn {

af::array lcnn_adapt(
  const af::array& state_memory, const af::array& reservoir_w, const lcnn_adaptation_config& cfg)
{
    // TODO add 32 bit variant
    if (state_memory.type() != af::dtype::f64)
        throw std::invalid_argument(
          "CUDA kernel for lcnn adapt is only supported for 64bit arrays.");

    // Evaluate input matrices.
    af::array prev_prev_state = state_memory(af::span, af::span, 2);
    prev_prev_state.eval();
    double* pprev_prev_state = prev_prev_state.device<double>();

    af::array prev_state = state_memory(af::span, af::span, 1);
    prev_state.eval();
    double* pprev_state = prev_state.device<double>();

    af::array curr_state = state_memory(af::span, af::span, 0);
    curr_state.eval();
    double* pcurr_state = curr_state.device<double>();

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
    dim3 grid(
      (curr_state.dims(0) + block.x - 1) / block.x, (curr_state.dims(1) + block.y - 1) / block.y);
    int perimeter_bytes =
      sizeof(double) * (block.x + reservoir_w.dims(2) - 1) * (block.y + reservoir_w.dims(3) - 1);
    lcnn_adapt_kernel<<<grid, block, 2 * perimeter_bytes, af_cuda_stream>>>(
      pprev_prev_state, pprev_state, pcurr_state, preservoir_w, poutput, curr_state.dims(0),
      curr_state.dims(1), reservoir_w.dims(2), reservoir_w.dims(3), cfg.learning_rate,
      cfg.weight_leakage);
    if (cudaError_t err = cudaPeekAtLastError(); err != cudaSuccess)
        throw std::runtime_error{
          "CUDA Runtime Error in LCNN step kernel launch: " + std::string{cudaGetErrorString(err)}};

    // Give matrices back to ArrayFire.
    prev_prev_state.unlock();
    prev_state.unlock();
    curr_state.unlock();
    reservoir_w.unlock();
    output.unlock();

    return output;
}

}  // namespace esn