#include "../lcnn_step.hpp"

#include <gtest/gtest.h>

TEST(LcnnStepCudaTest, ConstTest)
{
    int state_height = 200;
    int state_width = 200;
    for (int kernel_height = 3; kernel_height < 10; kernel_height += 2) {
        for (int kernel_width = 3; kernel_width < 10; kernel_width += 2) {
            af::array state = af::constant(3, state_height, state_width, af::dtype::f64);
            af::array reservoir_w = af::constant(
              5, state_height, state_width, kernel_height, kernel_width, af::dtype::f64);
            af::array new_state = lcnn_step(state, reservoir_w);
            af::array expected = af::constant(
              kernel_height * kernel_width * 3 * 5, state_height, state_width, af::dtype::f64);
            ASSERT_TRUE(af::allTrue<bool>(af::abs(new_state - expected) < 1e-18));
        }
    }
}