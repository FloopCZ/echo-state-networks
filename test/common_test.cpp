#include "common_test.hpp"

#include "../lcnn_step.hpp"

#include <arrayfire.h>
#include <gtest/gtest.h>

void const_test()
{
    af::info();
    int state_height = 100;
    int state_width = 200;
    af::array state = af::constant(3, state_height, state_width, af::dtype::f64);
    for (int kernel_height = 3; kernel_height < 30; kernel_height += 2) {
        for (int kernel_width = 3; kernel_width < 30; kernel_width += 2) {
            af::array reservoir_w = af::constant(
              5, state_height, state_width, kernel_height, kernel_width, af::dtype::f64);
            af::array new_state = lcnn_step(state, reservoir_w);
            af::array expected = af::constant(3 * 5, state_height, state_width, af::dtype::f64);
            ASSERT_TRUE(af::allTrue<bool>(af::abs(new_state - expected) < 1e-12));
        }
    }
}

static void test_step(int state_height, int state_width, int kernel_height, int kernel_width)
{
    af::array state = af::randu(state_height, state_width, af::dtype::f64);
    af::array reservoir_w =
      af::randu(state_height, state_width, kernel_height, kernel_width, af::dtype::f64);
    af::array new_state = lcnn_step(state, reservoir_w);
    for (int i = 0; i < state_height; ++i) {
        for (int j = 0; j < state_width; ++j) {
            double sum = 0;
            for (int k = -kernel_height / 2; k <= kernel_height / 2; ++k) {
                for (int l = -kernel_width / 2; l <= kernel_width / 2; ++l) {
                    af::array s = state(
                      ((i + k) % state_height + state_height) % state_height,
                      ((j + l) % state_width + state_width) % state_width);
                    af::array w = reservoir_w(i, j, k + kernel_height / 2, l + kernel_width / 2);
                    sum += w.scalar<double>() * s.scalar<double>();
                }
            }
            double mean = sum / kernel_height / kernel_width;
            ASSERT_NEAR(mean, new_state(i, j).scalar<double>(), 1e-12);
        }
    }
}

void random_stress_test()
{
    af::info();
    for (int state_height = 3; state_height < 60; state_height += 23) {
        for (int state_width = 5; state_width < 60; state_width += 29) {
            for (int kernel_height = 3; kernel_height < 8; kernel_height += 2) {
                for (int kernel_width = 3; kernel_width < 8; kernel_width += 2) {
                    test_step(state_height, state_width, kernel_height, kernel_width);
                }
            }
        }
    }
}