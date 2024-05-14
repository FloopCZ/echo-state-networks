#pragma once

#include <arrayfire.h>
#include <cmath>
#include <random>

namespace esn {

inline const double E = std::exp(1);

using prng_t = std::mt19937_64;
static thread_local std::random_device global_rd;
static thread_local prng_t global_prng{global_rd()};

constexpr long DEFAULT_SEED = 1000003L;

inline void reseed(prng_t& prng, long bias = 0)
{
    std::seed_seq seq({prng() + bias, prng() + bias, prng() + bias});
    prng.seed(seq);
}

/// Properly seed a random generator even with a single number.
///
/// If the seed is 0, the generator is seeded via random_device.
inline long set_seed(prng_t& prng, long seed)
{
    if (seed == 0) seed = global_rd();
    std::seed_seq seq({seed, seed * 2 + 2, seed * 3 + 3});
    prng.seed(seq);
    return seed;
}

inline long set_global_seed(long seed)
{
    return set_seed(global_prng, seed);
}

struct optimization_status_t {
    double progress;
};

}  // namespace esn
