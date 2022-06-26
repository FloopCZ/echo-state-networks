#pragma once

#include <arrayfire.h>
#include <random>

namespace esn {

static thread_local std::random_device global_rd;
static thread_local std::mt19937 global_prng{global_rd()};

/// The information returned by the esn feed().
struct feed_result_t {
    /// The array of intermediate states.
    af::array states;
    /// The array of intermediate outputs.
    af::array outputs;
};

}  // namespace esn
