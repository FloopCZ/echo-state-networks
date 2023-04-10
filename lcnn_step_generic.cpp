#include "lcnn_step.hpp"

// The shifting method creates a long JIT function which has to be limited
// to avoid a failed nvcc compilation.
constexpr int SHIFT_STEP_MAX_JIT_SIZE = 20;

af::array lcnn_step(const af::array& state, const af::array& reservoir_w)
{
    af::eval(state);
    af::eval(reservoir_w);
    int kernel_height = reservoir_w.dims(2);
    int kernel_width = reservoir_w.dims(3);
    af::array new_state = af::constant(0, state.dims(), state.type());
    // for each kernel coordinate
    for (int i = 0; i < kernel_height; ++i) {
        for (int j = 0; j < kernel_width; ++j) {
            af::array shifted_state =
              af::shift(state, -i + kernel_height / 2, -j + kernel_width / 2);
            af::array channel_state =
              reservoir_w(af::span, af::span, i, j) * std::move(shifted_state);
            // Multiply the kernel channel and the activations
            // from the periodic state matrix. Append it to the new_state of the
            // corresponding neurons.
            new_state += std::move(channel_state);
            if ((i * kernel_height + j) % SHIFT_STEP_MAX_JIT_SIZE == 0) af::eval(new_state);
        }
    }
    af::eval(new_state);
    return new_state;
}