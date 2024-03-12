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

}  // namespace esn
