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
    int kernel_size = K * L;

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
    double* perimeter_presynaptic_state = shm;
    double* perimeter_presynaptic_diff = shm + perimeter_size;
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
        perimeter_presynaptic_state[perimeter_idx] = prev_state[input_i + N * input_j];
        perimeter_presynaptic_diff[perimeter_idx] =
          prev_state[input_i + N * input_j] - prev_prev_state[input_i + N * input_j];
    }

    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= N || j >= M) return;

    double postsynaptic_state = curr_state[i + N * j];
    double postsynaptic_diff = curr_state[i + N * j] - prev_state[i + N * j];

    double abs_sum_before = 0.;
    double abs_sum_after = 0.;
    for (int l = 0; l < L; ++l) {
        int perimeter_j = threadIdx.y + l;
        for (int k = 0; k < K; ++k) {
            // if (j == 0 && k == 0) continue;
            int perimeter_i = threadIdx.x + k;
            int perimeter_idx = perimeter_i + perimeter_height * perimeter_j;
            int reservoir_idx = i + N * j + N * M * k + N * M * K * l;
            double presynaptic_state = perimeter_presynaptic_state[perimeter_idx];
            double presynaptic_diff = perimeter_presynaptic_diff[perimeter_idx];
            double weight = reservoir_w[reservoir_idx];
            abs_sum_before += abs(weight);
            double learning_strength = 1;
            double delta =
              pow(presynaptic_diff * postsynaptic_diff, 3) * learning_strength * learning_rate;
            if (i == 1 && j == 7) {
                printf("i,j,k,l = %d,%d,%d,%d\n", i, j, k, l);
                printf("weight %.10f\n", weight);
                printf("delta %.10f\n", delta);
                printf("presynatpic diff i,j = %d,%d, value = %.10f\n", i, j, presynaptic_diff);
                printf("postsynatpic diff i,j = %d,%d, value = %.10f\n", i, j, postsynaptic_diff);
                printf("presynatpic state i,j = %d,%d, value = %.10f\n", i, j, presynaptic_state);
                printf("postsynatpic state i,j = %d,%d, value = %.10f\n", i, j, postsynaptic_state);
                printf("\n");
            }
            weight += delta;
            abs_sum_after += abs(weight);
            output[reservoir_idx] = weight;
        }
    }

    if (i == 1 && j == 7) {
        printf("abs before %.2f\n", abs_sum_before);
        printf("abs after %.2f\n", abs_sum_after);
    }

    double norm_factor = abs_sum_before / abs_sum_after;
    for (int l = 0; l < L; ++l) {
        // int perimeter_j = threadIdx.y + l;
        for (int k = 0; k < K; ++k) {
            // if (j == 0 && k == 0) continue;
            // int perimeter_i = threadIdx.x + k;
            // int perimeter_idx = perimeter_i + perimeter_height * perimeter_j;
            int reservoir_idx = i + N * j + N * M * k + N * M * K * l;
            double weight = output[reservoir_idx];
            output[reservoir_idx] = weight * norm_factor * (1. - weight_leakage);
        }
    }
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