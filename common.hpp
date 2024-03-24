#pragma once

#include <arrayfire.h>
#include <cassert>
#include <optional>
#include <random>

namespace esn {

using prng_t = std::mt19937_64;
static thread_local std::random_device global_rd;
static thread_local prng_t global_prng{global_rd()};

constexpr long DEFAULT_SEED = 1000003L;

inline long set_global_seed(long seed)
{
    if (seed == 0) seed = global_rd();
    global_prng.seed(seed);
    return seed;
}

struct optimization_status_t {
    double progress;
};

}  // namespace esn
