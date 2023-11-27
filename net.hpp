#pragma once

// Virtual base class for a network. //

#include "common.hpp"
#include "data_map.hpp"

#include <arrayfire.h>
#include <cassert>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace esn {

constexpr af::dtype DEFAULT_AF_DTYPE = af::dtype::f64;

class net_base {
public:
    struct input_t {
        data_map input;
        data_map feedback;
        data_map desired;
    };

    struct on_state_change_data {
        af::array state;
        input_t input;
        af::array output;
        std::optional<std::string> event;
    };

    using on_state_change_callback_t = std::function<void(net_base&, on_state_change_data)>;

protected:
    std::vector<on_state_change_callback_t> on_state_change_callbacks_;
    std::optional<std::string> event_;

public:
    virtual void step(data_map step_input, data_map step_feedback, data_map step_desired) = 0;

    virtual void event(const std::string& event)
    {
        event_ = event;
    }

    virtual feed_result_t feed(const input_t& input) = 0;

    virtual std::tuple<feed_result_t, train_result_t> train(const input_t& input) = 0;

    virtual train_result_t train(const feed_result_t& data) = 0;

    virtual const af::array& state() const = 0;

    virtual void state(af::array new_state) = 0;

    virtual const std::set<std::string>& input_names() const = 0;

    virtual const std::set<std::string>& output_names() const = 0;

    virtual double neuron_ins() const = 0;

    virtual void learning_rate(double) = 0;

    virtual void random_noise(bool) = 0;

    /// Add a callback to call when the state has changed.
    virtual void add_on_state_change(on_state_change_callback_t fnc)
    {
        on_state_change_callbacks_.push_back(std::move(fnc));
    }

    virtual std::unique_ptr<net_base> clone() const = 0;

    virtual ~net_base() = default;
};

}  // end namespace esn
