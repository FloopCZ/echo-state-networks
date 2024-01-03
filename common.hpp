#pragma once

#include <arrayfire.h>
#include <cassert>
#include <optional>
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
    /// The array of intermediate desired values.
    std::optional<af::array> desired;
};

/// The information returned by the esn train().
struct train_result_t {
    /// The array of intermediate predictors.
    af::array predictors;
    /// The output weights.
    af::array output_w;
};

inline feed_result_t concatenate(const feed_result_t& a, const feed_result_t& b)
{
    if (a.states.isempty() && a.outputs.isempty() && !a.desired) return b;

    feed_result_t c;
    assert(a.states.type() == b.states.type());
    assert(a.outputs.type() == b.outputs.type());
    assert(a.outputs.numdims() <= 2);
    assert(b.outputs.numdims() <= 2);
    assert(a.desired.has_value() == b.desired.has_value());
    if (a.states.numdims() == 2) {
        // 1d state
        assert(a.states.dims(0) == b.states.dims(0));
        assert(a.states.dims(1) == a.outputs.dims(1));
        c.states = af::join(1, a.states, b.states);
    } else if (a.states.numdims() == 3) {
        // 2d state
        assert(a.states.dims(0) == b.states.dims(0));
        assert(a.states.dims(1) == b.states.dims(1));
        assert(a.states.dims(2) == a.outputs.dims(1));
        c.states = af::join(2, a.states, b.states);
    } else {
        throw std::runtime_error(
          "Unsupported dimensionality of states " + std::to_string(a.states.numdims()) + ".");
    }
    c.outputs = af::join(1, a.outputs, b.outputs);
    if (a.desired.has_value() && b.desired.has_value()) {
        assert(a.desired->type() == b.desired->type());
        assert(a.desired->numdims() <= 2);
        assert(a.desired->dims() == a.outputs.dims());
        assert(b.desired->numdims() <= 2);
        assert(b.desired->dims() == b.outputs.dims());
        c.desired = af::join(1, *a.desired, *b.desired);
    }
    return c;
}

}  // namespace esn
