#include "common_test.hpp"

#include <gtest/gtest.h>

TEST(LcnnStepCudaTest, ConstTest)
{
    return const_test();
}

TEST(LcnnStepCudaTest, RandStressTest)
{
    return random_stress_test();
}