#include "lcnn_step.hpp"
#include "lcnn_step_generic.hpp"

af::array lcnn_step(const af::array& state, const af::array& reservoir_w)
{
    return lcnn_step_generic(state, reservoir_w);
}

// Custom naive CPU implementation does not provide any speedup even for a single thread.
// Salute, ArrayFire.
//
// af::array lcnn_step(const af::array& state, const af::array& reservoir_w)
// {
//     // TODO add 32 bit variant
//     if (state.type() != af::dtype::f64)
//         throw std::invalid_argument(
//           "CUDA kernel for lcnn step is only supported for 64bit arrays.");

//     int state_height = reservoir_w.dims(0);
//     int state_width = reservoir_w.dims(1);
//     int kernel_height = reservoir_w.dims(2);
//     int kernel_width = reservoir_w.dims(3);

//     af::eval(state);
//     double* pstate = state.device<double>();
//     af::eval(reservoir_w);
//     double* preservoir_w = reservoir_w.device<double>();
//     af::array output = af::constant(0, state.dims(), state.type());
//     af::eval(output);
//     double* poutput = output.device<double>();

//     long kernel_radius_height = kernel_height / 2;
//     long kernel_radius_width = kernel_width / 2;
//     long k_coef = state_height * state_width;
//     long l_coef = k_coef * kernel_height;
//     for (int i = 0; i < state_height; ++i) {
//         for (int j = 0; j < state_width; ++j) {
//             long state_idx = i + state_height * j;
//             for (int k = 0; k < kernel_height; ++k) {
//                 long input_i = (i - kernel_radius_height + k) % state_height;
//                 input_i = (input_i % state_height + state_height) % state_height;
//                 for (int l = 0; l < kernel_width; ++l) {
//                     long input_j = (j - kernel_radius_width + l) % state_width;
//                     input_j = (input_j % state_width + state_width) % state_width;
//                     long input_idx = input_i + state_height * input_j;
//                     long reservoir_w_idx = i + state_height * j + k_coef * k + l_coef * l;
//                     poutput[state_idx] += preservoir_w[reservoir_w_idx] * pstate[input_idx];
//                 }
//             }
//         }
//     }
//     state.unlock();
//     reservoir_w.unlock();
//     output.unlock();

//     return output;
// }