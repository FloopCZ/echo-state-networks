#pragma once

#include <arrayfire.h>
#include <cassert>
#include <optional>
#include <random>

namespace esn {

static thread_local std::random_device global_rd;
static thread_local std::mt19937 global_prng{global_rd()};

}  // namespace esn
