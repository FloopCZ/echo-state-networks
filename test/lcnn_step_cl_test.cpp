#include "common_test.hpp"

#include <gtest/gtest.h>

TEST(LcnnStepClTest, ConstTest)
{
    return const_test();
}

TEST(LcnnStepClTest, RandStressTest)
{
    return random_stress_test();
}