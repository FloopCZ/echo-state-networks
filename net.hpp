#pragma once

// Virtual base class for a network. //

#include "common.hpp"

#include <arrayfire.h>
#include <cassert>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <vector>

namespace esn {

class net_base {
public:
    struct on_state_change_data {
        af::array state;
        af::array input;
        af::array output;
        std::optional<af::array> desired;
    };

    using on_state_change_callback_t = std::function<void(net_base&, on_state_change_data)>;

protected:
    std::vector<on_state_change_callback_t> on_state_change_callbacks_;

public:
    virtual void step(
      const af::array& input,
      const std::optional<af::array>& feedback = std::nullopt,
      const std::optional<af::array>& desired = std::nullopt) = 0;

    /// Perform a single step while putting the same value to all inputs.
    /// \param input The input value.
    /// \param feedback Optional feedback value.
    /// \param desired Optional desired value.
    virtual void step_constant(
      double input,
      std::optional<double> feedback = std::nullopt,
      std::optional<double> desired = std::nullopt)
    {
        assert(!feedback || !desired);
        af::array af_input = af::constant(input, n_ins(), state().type());
        std::optional<af::array> af_feedback;
        if (feedback) af_feedback = af::constant(*feedback, n_outs(), state().type());
        std::optional<af::array> af_desired;
        if (desired) af_desired = af::constant(*desired, n_outs(), state().type());
        return step(af_input, af_feedback, af_desired);
    }

    virtual feed_result_t feed(
      const af::array& input,
      const std::optional<af::array>& feedback = std::nullopt,
      const std::optional<af::array>& desired = std::nullopt) = 0;

    virtual feed_result_t train(const af::array& input, const af::array& desired) = 0;

    virtual feed_result_t
    loop(long n_steps, const std::optional<af::array>& desired = std::nullopt) = 0;

    virtual const af::array& state() const = 0;

    virtual void state(af::array new_state) = 0;

    virtual long n_ins() const = 0;

    virtual long n_outs() const = 0;

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
