#pragma once

// Virtual base class for a network. //

#include "common.hpp"
#include "data_map.hpp"

#include <arrayfire.h>
#include <cassert>
#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace esn {

constexpr af::dtype DEFAULT_AF_DTYPE = af::dtype::f64;

using input_transform_fn_t = std::function<data_map(const data_map&)>;
namespace fs = std::filesystem;

/// The input data.
struct input_t {
    data_map input;
    data_map feedback;
    data_map desired;
    input_transform_fn_t input_transform;
};

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

class net_base {
public:
    struct on_state_change_data {
        af::array state;
        input_t input;
        data_map output;
        std::optional<std::string> event;
    };

    using on_state_change_callback_t = std::function<void(net_base&, on_state_change_data)>;

protected:
    std::vector<on_state_change_callback_t> on_state_change_callbacks_;
    std::optional<std::string> event_;

public:
    virtual void step(
      const data_map& step_input,
      const data_map& step_feedback,
      const data_map& step_desired,
      input_transform_fn_t input_transform) = 0;

    virtual void event(const std::string& event)
    {
        event_ = event;
    }

    virtual feed_result_t feed(const input_t& input) = 0;

    virtual train_result_t train(const input_t& input) = 0;

    virtual train_result_t train(feed_result_t data) = 0;

    virtual void clear_feedback() = 0;

    virtual const af::array& state() const = 0;

    virtual void state(af::array new_state) = 0;

    virtual const std::set<std::string>& input_names() const = 0;

    virtual const std::set<std::string>& output_names() const = 0;

    virtual double neuron_ins() const = 0;

    virtual void random_noise(bool) = 0;

    virtual void learning(bool) = 0;

    virtual void reset() = 0;

    /// Add a callback to call when the state has changed.
    virtual void add_on_state_change(on_state_change_callback_t fnc)
    {
        on_state_change_callbacks_.push_back(std::move(fnc));
    }

    virtual std::unique_ptr<net_base> clone() const = 0;

    virtual void save(const fs::path& dir) = 0;

    virtual ~net_base() = default;
};

}  // end namespace esn
