#include "common_test.hpp"

#include <gtest/gtest.h>

TEST(LcnnStepCpuTest, ConstTest)
{
    return const_test();
}

TEST(LcnnStepCpuTest, RandStressTest)
{
    return random_stress_test();
}