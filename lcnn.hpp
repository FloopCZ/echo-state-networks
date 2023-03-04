#pragma once

// Locally connected Echo state network class and training functions. //

#include "common.hpp"
#include "net.hpp"
#include "simple_esn.hpp"

#include <arrayfire.h>
#include <boost/program_options.hpp>
#include <cassert>
#include <random>

namespace esn {

// The size of the height or width of the kernel above which the matrix multiplication
// will be used instead of unwrapping.
// For the matrix sizes we use, the matrix multiplication is more efficient, so
// the threshold is intentionally set to zero.
constexpr int KERNEL_SIZE_MATMUL_THRESHOLD = 0;

namespace po = boost::program_options;

/// Locally connected network configuration.
struct lcnn_config {
    /// The initial state of the network of size [state_height, state_width].
    af::array init_state;
    /// The reservoir weights of size [state_height, state_width, kernel_height, kernel_width].
    af::array reservoir_w;
    /// The reservoir weights of size [state_height*state_width, state_height*state_width].
    /// This field is optional and intended for performance optimization.
    af::array reservoir_w_full;
    /// The reservoir biases of size [state_height, state_width].
    af::array reservoir_b;
    /// The input weights of size [state_height, state_width, n_ins].
    af::array input_w;
    /// The feedback weights of size [state_height, state_width, n_outs].
    af::array feedback_w;

    // Learning is not available at the moment.
    // /// If it is set to 0.0, the network does not learn.
    // double learning_rate = 0;
    // /// The multiplier of the difference between the neuron's state and the
    // /// exponential moving average.
    // double learning_rate_ema = 0;
    // /// The multiplier for the difference between two potentials (post - pre).
    // double learning_rate_diff = 0;
    // /// The constant towards which are the neurons biased.
    // double learning_bias = 0;
    // /// The rate by which do the weights change to get neurons towards the bias.
    // double learning_rate_bias = 0;
    // /// The rate of exponential weight decay.
    // double weight_decay = 0;
    // /// The decay rate of the exponential moving average of the state used for weight learning.
    // double state_ema_decay = 0;
    /// The probability that a neuron will fire in every moment.

    double random_spike_prob = 0;
    /// The standard deviation of the random spike.
    double random_spike_std = 0;
    /// The standard deviation of the noise added to the potentials.
    double noise = 0;
    /// The leakage of the potential.
    double leakage = 1.0;

    lcnn_config() = default;
    lcnn_config(const po::variables_map& args)
    {
        // Learning is not available at the moment.
        // learning_rate      = args.at("lcnn.learning-rate").as<double>();
        // learning_rate_ema  = args.at("lcnn.learning-rate-ema").as<double>();
        // learning_rate_diff = args.at("lcnn.learning-rate-diff").as<double>();
        // learning_bias      = args.at("lcnn.learning-bias").as<double>();
        // learning_rate_bias = args.at("lcnn.learning-rate-bias").as<double>();
        // weight_decay       = args.at("lcnn.weight-decay").as<double>();
        // state_ema_decay    = args.at("lcnn.state-ema-decay").as<double>();

        random_spike_prob = args.at("lcnn.random-spike-prob").as<double>();
        random_spike_std = args.at("lcnn.random-spike-std").as<double>();
        noise = args.at("lcnn.noise").as<double>();
        leakage = args.at("lcnn.leakage").as<double>();
    }
};

/// Advanced Echo State Networks with various reservoir topologies.
template <af::dtype DType = af::dtype::f64>
class lcnn : public net_base {
protected:
    long n_ins_;
    long n_outs_;
    af::array state_delta_;  // working variable used during the step function
    af::array state_;
    af::array last_output_;  // the last output of the net
    af::array reservoir_w_;
    std::vector<std::vector<af::array>> kernel_channels_;
    af::array reservoir_w_full_;
    af::array reservoir_b_;
    af::array input_w_;
    af::array feedback_w_;
    af::array output_w_;
    bool force_matmul_;

    // Random engines.
    std::mt19937* prng_;
    af::randomEngine af_prng_;

    // Learning is not available at the moment.
    // af::array state_ema_;  // exponential moving average
    // double learning_rate_;
    // double learning_rate_ema_;
    // double learning_rate_diff_;
    // double learning_bias_;
    // double learning_rate_bias_;
    // double weight_decay_;
    // double state_ema_decay_;

    bool noise_enabled_;
    double random_spike_prob_;
    double random_spike_std_;
    double noise_;
    double leakage_;

    /// Return whether the step should be performed by matmul or by unwrapping.
    bool do_matmul_step() const
    {
        if (force_matmul_) return true;
        if (reservoir_w_.dims(2) >= KERNEL_SIZE_MATMUL_THRESHOLD) return true;
        if (reservoir_w_.dims(3) >= KERNEL_SIZE_MATMUL_THRESHOLD) return true;
        return false;
    }

    af::array update_via_weights_matmul_impl(const af::array& state) const
    {
        af::array new_state = af::matmul(reservoir_w_full_, af::flat(state));
        return af::moddims(std::move(new_state), state.dims());
    }

    /// Update the state matrix using masked matrix multiplication.
    virtual void update_via_weights_matmul()
    {
        state_delta_ = update_via_weights_matmul_impl(state_);
    }

    af::array update_via_weights_impl(const af::array& state)
    {
        int kernel_height = reservoir_w_.dims(2);
        int kernel_width = reservoir_w_.dims(3);

        af::array new_state = af::constant(0, state.dims(), state.type());
        // for each kernel coordinate
        for (int i = 0; i < kernel_height; ++i) {
            for (int j = 0; j < kernel_width; ++j) {
                af::array shifted_state =
                  af::shift(state, -i + kernel_height / 2, -j + kernel_width / 2);
                af::array channel_state = kernel_channels_.at(i).at(j) * std::move(shifted_state);
                // Multiply the kernel channel and the activations
                // from the periodic state matrix. Append it to the new_state of the
                // corresponding neurons.
                new_state += std::move(channel_state);
            }
        }

        return new_state;
    }

    /// Update the state matrix from the unwrapped weights and previous state.
    virtual void update_via_weights()
    {
        state_delta_ = update_via_weights_impl(state_);
    }

    /// Update the state matrix by adding the inputs.
    virtual void update_via_input(const af::array& input)
    {
        af::array input_w = af::moddims(input_w_, state_.dims(0) * state_.dims(1), n_ins_);
        af::array delta = af::matmul(std::move(input_w), input);
        state_delta_ += af::moddims(std::move(delta), state_.dims());
    }

    /// Update the state matrix by adding the feedback.
    virtual void update_via_feedback(const af::array& feedback)
    {
        af::array feedback_w = af::moddims(feedback_w_, state_.dims(0) * state_.dims(1), n_outs_);
        af::array delta = af::matmul(std::move(feedback_w), feedback);
        state_delta_ += af::moddims(std::move(delta), state_.dims());
    }

    /// Update the state matrix by adding the bias.
    virtual void update_via_bias()
    {
        state_delta_ += reservoir_b_;
    }

    /// Update the state matrix by applying the activation function.
    virtual void update_via_activation()
    {
        // Add spurious spikes.
        if (noise_enabled_ && random_spike_prob_ > 0.) {
            af::array spikes =
              2. * af::randu({state_.dims()}, DType, af_prng_) < random_spike_prob_ - 1.;
            spikes *= random_spike_std_;
            state_delta_ += spikes;
        }
        // Add noise to the states.
        if (noise_enabled_ && noise_ != 0.)
            state_delta_ *= 1. + af::randn({state_.dims()}, DType, af_prng_) * noise_;
        // Leak some potential.
        state_ *= 1. - leakage_;
        // Apply the activation function.
        state_ += af::tanh(std::move(state_delta_));
        // Update the state with the proper leakage.
        af::eval(state_);
    }

    // Learning is not available at the moment.
    // /// Update the weights.
    // ///
    // /// WARNING: This algorithm does not sucessfully learn yet.
    // virtual void update_weights(
    //   const af::array& /* unwrapped_weights */,
    //   const af::array& unwrapped_old_state)
    // {
    //     int kernel_height = reservoir_w_.dims(2);
    //     int kernel_width = reservoir_w_.dims(3);
    //     // calculate presynaptic and postsynaptic states for each connection
    //     af::array presynaptic_state = unwrapped_old_state;
    //     af::array postsynaptic_state =
    //       af::tile(af::flat(state_), 1, kernel_height * kernel_width);
    //     // match state_ema_ to each of the connections
    //     af::array state_ema =
    //       af::tile(af::flat(state_ema_), 1, kernel_height * kernel_width);

    //     // Update the weights according to the difference from ema.
    //     af::array weight_delta =
    //       learning_rate_ema_ * af::pow(postsynaptic_state - state_ema, 3.);
    //     // Update the weights according to the difference between pre and
    //     // postsynaptic state.
    //     weight_delta +=
    //       learning_rate_diff_ * af::pow(postsynaptic_state - presynaptic_state, 3.);
    //     // Update the weights towards the bias.
    //     weight_delta +=
    //       learning_rate_bias_ * af::pow(postsynaptic_state - learning_bias_, 3.);

    //     // Reshape the weight deltas back to the reservoir_w_ shape and update.
    //     weight_delta = af::moddims(std::move(weight_delta), reservoir_w_.dims());
    //     reservoir_w_ += learning_rate_ * std::move(weight_delta);
    //     // decay the weights
    //     reservoir_w_ *= 1. - weight_decay_;
    //     af::eval(reservoir_w_);
    // }

    /// Update the last output of the network after having a new state.
    virtual void update_last_output(const std::optional<af::array>& feedback = std::nullopt)
    {
        if (feedback) {
            last_output_ = *feedback;
        } else {
            af::array flat_state_1 = af_utils::add_ones(af::flat(state_), 0);
            last_output_ = af::clamp(af::matmul(output_w_, flat_state_1), -1., 1.);
        }
        assert(last_output_.dims() == (af::dim4{n_outs_}));
    }

public:
    lcnn() = default;

    /// Locally connected echo state network constructor.
    lcnn(lcnn_config cfg, std::mt19937& prng)
      : n_ins_{cfg.input_w.dims(2)}
      , n_outs_{cfg.feedback_w.dims(2)}
      , last_output_{af::constant(0, n_outs_, DType)}
      , output_w_{af::constant(af::NaN, n_outs_, cfg.init_state.elements() + 1, DType)}
      , force_matmul_{false}
      , prng_{&prng}
      , af_prng_{AF_RANDOM_ENGINE_DEFAULT, prng_->operator()()}

      // Learning is not available at the moment.
      // , learning_rate_       {cfg.learning_rate}
      // , learning_rate_ema_   {cfg.learning_rate_ema}
      // , learning_rate_diff_  {cfg.learning_rate_diff}
      // , learning_bias_       {cfg.learning_bias}
      // , learning_rate_bias_  {cfg.learning_rate_bias}
      // , weight_decay_        {cfg.weight_decay}
      // , state_ema_decay_     {cfg.state_ema_decay}

      , noise_enabled_{true}
      , random_spike_prob_{cfg.random_spike_prob}
      , random_spike_std_{cfg.random_spike_std}
      , noise_{cfg.noise}
      , leakage_{cfg.leakage}
    {
        state(std::move(cfg.init_state));
        assert(cfg.reservoir_w.isempty() ^ cfg.reservoir_w_full.isempty());
        if (!cfg.reservoir_w.isempty()) reservoir_weights(std::move(cfg.reservoir_w));
        if (!cfg.reservoir_w_full.isempty())
            reservoir_weights_full(std::move(cfg.reservoir_w_full));
        reservoir_biases(std::move(cfg.reservoir_b));
        input_weights(std::move(cfg.input_w));
        feedback_weights(std::move(cfg.feedback_w));
    }

    /// Perform a single step with a single input.
    /// \param input The input value.
    /// \param feedback The teacher-forced feedback to be used instead
    ///                 of the network's output.
    /// \param desired The desired output. This is only used for callbacks.
    ///                This is only allowed if no feedback is provided.
    ///                Has to be of size [n_outs].
    void step(
      const af::array& input,
      const std::optional<af::array>& feedback = std::nullopt,
      const std::optional<af::array>& desired = std::nullopt) override
    {
        assert(!desired || !feedback);
        assert(input.dims() == af::dim4{n_ins_});
        assert((!feedback || feedback->dims() == af::dim4{n_outs_}));
        assert((!feedback || af::allTrue<bool>(feedback.value() >= -1. && feedback.value() <= 1.)));
        assert((!desired || desired->dims() == af::dim4{n_outs_}));
        assert((!desired || af::allTrue<bool>(desired.value() >= -1. && desired.value() <= 1.)));

        // Update the internal state.
        // Perform matrix multiplication instead of state unwrapping for large kernels.
        if (do_matmul_step()) {
            update_via_weights_matmul();
        } else {
            // Use state unwrapping for small kernels
            update_via_weights();
        }

        // add input
        update_via_input(input);

        // add feedback
        update_via_feedback(last_output_);

        // add bias
        update_via_bias();

        // activation function
        update_via_activation();

        // Learning is not available at the moment.
        // if (learning_rate_ != 0) {
        //     // update state exponential moving average
        //     update_state_ema();
        //     // update the weights
        //     update_weights(unwrapped_weights, unwrapped_state);
        // }

        // update last output after we have a new state
        update_last_output(feedback);

        // Call the registered callback functions.
        for (on_state_change_callback_t& fnc : on_state_change_callbacks_) {
            on_state_change_data data = {
              .state = state_, .input = input, .output = last_output_, .desired = desired};
            fnc(*this, std::move(data));
        }
    }

    /// Perform multiple steps with multiple input seqences.
    /// \param inputs Input sequence of dimensions [n_ins, time].
    /// \param feedback The desired output sequences to be teacher-forced into the net.
    ///                 Needs to have dimensions [n_outs, time]
    /// \param desired The desired output. This is only used for callbacks.
    ///                This is only allowed if no feedback is provided.
    ///                Has to be of size [n_outs, time].
    /// \return The array of intermediate states of dimensions [state_height, state_width, time]
    ///         and the array of intermediate outputs of dimensions [n_outs, time].
    feed_result_t feed(
      const af::array& input,
      const std::optional<af::array>& feedback = std::nullopt,
      const std::optional<af::array>& desired = std::nullopt) override
    {
        assert(!desired || !feedback);
        assert(input.type() == DType);
        assert(input.numdims() == 2);
        assert(input.dims(0) == n_ins_);
        assert(input.dims(1) > 0);
        assert(!feedback || feedback->type() == DType);
        assert(!feedback || feedback->numdims() <= 2);
        assert(!feedback || feedback->dims(0) == n_outs_);
        assert(!feedback || feedback->dims(1) == input.dims(1));
        assert(!feedback || af::allTrue<bool>(af::abs(feedback.value()) <= 1.));
        assert(!desired || desired->type() == DType);
        assert(!desired || desired->numdims() <= 2);
        assert(!desired || desired->dims(0) == n_outs_);
        assert(!desired || desired->dims(1) == input.dims(1));
        assert(!desired || af::allTrue<bool>(af::abs(desired.value()) <= 1.));
        feed_result_t result;
        result.states = af::array(state_.dims(0), state_.dims(1), input.dims(1), DType);
        result.outputs = af::array(n_outs_, input.dims(1), DType);
        for (long i = 0; i < input.dims(1); ++i) {
            std::optional<af::array> feedback_i;
            if (feedback) feedback_i = feedback.value()(af::span, i);
            std::optional<af::array> desired_i;
            if (desired) desired_i = desired.value()(af::span, i);
            step(input(af::span, i), feedback_i, desired_i);
            result.states(af::span, af::span, i) = state_;
            result.outputs(af::span, i) = last_output_;
        }
        return result;
    }

    /// Train the network on the given sequence.
    /// \param input Input sequence of dimensions [n_ins, time].
    /// \param desired The desired output sequences. Those are also teacher-forced into the net.
    ///                Needs to have dimensions [n_outs, time]
    feed_result_t train(const af::array& input, const af::array& desired) override
    {
        assert(input.type() == DType);
        assert(input.numdims() == 2);
        assert(input.dims(0) == n_ins_);
        assert(input.dims(1) > 0);
        assert(desired.type() == DType);
        assert(desired.numdims() <= 2);
        assert(desired.dims(0) == n_outs_);
        assert(desired.dims(1) == input.dims(1));
        assert(af::allTrue<bool>(af::abs(desired) <= 1));
        feed_result_t feed_result = feed(input, desired);
        af::array states = af::moddims(feed_result.states, state_.elements(), input.dims(1));
        output_w_ = af_utils::lstsq_train(states.T(), desired.T()).T();
        assert(output_w_.dims() == (af::dim4{n_outs_, state_.elements() + 1}));
        return feed_result;
    }

    /// Perform multiple steps using self's output as input.
    /// \param n_steps The number of steps to take.
    /// \param desired The desired output. This is used only for callbacks.
    ///                Needs to have dimensions [n_outs, time]
    /// \return The same as \ref feed().
    feed_result_t
    loop(long n_steps, const std::optional<af::array>& desired = std::nullopt) override
    {
        assert(n_ins_ == n_outs_);
        assert(n_steps > 0);
        assert(!desired || desired->type() == DType);
        assert(!desired || desired->numdims() <= 2);
        assert(!desired || desired->dims(0) == n_outs_);
        assert(!desired || desired->dims(1) == n_steps);
        assert(!desired || af::allTrue<bool>(af::abs(desired.value()) <= 1.));
        feed_result_t result;
        result.states = af::array(state_.dims(0), state_.dims(1), n_steps, DType);
        result.outputs = af::array(n_outs_, n_steps, DType);
        for (long i = 0; i < n_steps; ++i) {
            std::optional<af::array> desired_i;
            if (desired) desired_i = desired.value()(af::span, i);
            // Create a shallow copy of `last_output_` so that when step() changes it,
            // it's `input` parameter does not change (it is taken as const ref).
            step(af::array(last_output_), std::nullopt, desired_i);
            result.states(af::span, af::span, i) = state_;
            result.outputs(af::span, i) = last_output_;
        }
        return result;
    }

    /// Get the current state of the network.
    const af::array& state() const override
    {
        return state_;
    }

    /// Set the current state of the network.
    void state(af::array new_state) override
    {
        assert(new_state.type() == DType);
        assert(new_state.numdims() == 2);
        state_ = std::move(new_state);
    }

    // Learning is not available at the moment.
    // /// Update the state exponential moving average using the current state.
    // void update_state_ema()
    // {
    //     if (state_ema_.isempty()) {
    //         state_ema_ = state_;
    //     } else {
    //         state_ema_ = (1. - state_ema_decay_) * state_ema_ + state_ema_decay_ * state_;
    //     }
    // }

    /// The number of inputs.
    long n_ins() const override
    {
        return n_ins_;
    }

    /// The number of outputs.
    long n_outs() const override
    {
        return n_outs_;
    }

    /// The average number of inputs to a neuron.
    double neuron_ins() const override
    {
        if (force_matmul_) {
            af::array in = af::sum(reservoir_w_full_ != 0, 1);
            return af::mean<double>(in);
        }
        af::array in = af::sum(af::sum(reservoir_w_ != 0, 3), 2);
        return af::mean<double>(in);
    }

    /// Set the input weights of the network.
    ///
    /// The shape has to be [state.dims(0), state.dims(1), n_ins].
    void input_weights(af::array new_weights)
    {
        assert(new_weights.type() == DType);
        assert((new_weights.dims() == af::dim4{state_.dims(0), state_.dims(1), n_ins_}));
        input_w_ = std::move(new_weights);
    }

    /// Get the input weights of the network.
    const af::array& input_weights() const
    {
        return input_w_;
    }

    /// Set the feedback weights of the network.
    ///
    /// The shape has to be [state.dims(0), state.dims(1), n_outs].
    void feedback_weights(af::array new_weights)
    {
        assert(new_weights.type() == DType);
        assert((new_weights.dims() == af::dim4{state_.dims(0), state_.dims(1), n_outs_}));
        feedback_w_ = std::move(new_weights);
    }

    /// Get the feedback weights of the network.
    const af::array& feedback_weights() const
    {
        return feedback_w_;
    }

    /// Set the reservoir weights of the network.
    ///
    /// Also initializes the fully connected reservoir matrix.
    ///
    /// The shape has to be [state_height, state_width, kernel_height, kernel_width].
    void reservoir_weights(af::array new_weights)
    {
        assert(new_weights.type() == DType);
        assert(new_weights.numdims() == 4);
        assert(new_weights.dims(0) == state_.dims(0));
        assert(new_weights.dims(1) == state_.dims(1));
        assert(new_weights.dims(2) % 2 == 1);
        assert(new_weights.dims(3) % 2 == 1);
        reservoir_w_ = std::move(new_weights);
        force_matmul_ = false;

        // Precalculate kernel channels.
        // Channel is the 2D matrix made only of the kernel
        // elements on the fixed coordinate i, j for each neuron.
        // It has the same shape as the state.
        int kernel_height = reservoir_w_.dims(2);
        int kernel_width = reservoir_w_.dims(3);
        kernel_channels_.resize(kernel_height);
        for (int i = 0; i < kernel_height; ++i) {
            for (int j = 0; j < kernel_width; ++j) {
                kernel_channels_.at(i).push_back(reservoir_w_(af::span, af::span, i, j));
                kernel_channels_.at(i).at(j).eval();
            }
        }

        // Precalculate the fully connected weight matrix.
        if (do_matmul_step()) {
            int state_height = reservoir_w_.dims(0);
            int state_width = reservoir_w_.dims(1);

            // Convert the reservoir matrices on host for performance.
            std::vector<double> reservoir_w(
              state_height * state_width * kernel_height * kernel_width);
            reservoir_w_.host(reservoir_w.data());

            std::vector<double> reservoir_w_full(
              state_height * state_width * state_height * state_width);

            for (int i = 0; i < state_height; ++i) {
                for (int j = 0; j < state_width; ++j) {
                    for (int k = 0; k < kernel_height; ++k) {
                        for (int l = 0; l < kernel_width; ++l) {
                            int from_i = (i + k - kernel_height / 2 + state_height) % state_height;
                            int from_j = (j + l - kernel_width / 2 + state_width) % state_width;
                            int full_index = i + j * state_height
                              + (from_i + from_j * state_height) * state_height * state_width;
                            int sparse_index = i + j * state_height + k * state_height * state_width
                              + l * state_height * state_width * kernel_height;
                            reservoir_w_full[full_index] = reservoir_w[sparse_index];
                        }
                    }
                }
            }

            reservoir_weights_full(
              af::array{state_.elements(), state_.elements(), reservoir_w_full.data()}.as(DType));

// Check that the matmul style and kernel style step are the same.
#ifndef NDEBUG
            af::array rand_state = af::randu({state_.dims()}, DType, af_prng_) / state_.elements();
            af::array state_matmul = update_via_weights_matmul_impl(rand_state);
            af::array state_wrap = update_via_weights_impl(rand_state);
            assert(af_utils::almost_equal(state_matmul, state_wrap, 1e-12));
#endif
        }
    }

    /// Set the fully connected reservoir weights of the network.
    ///
    /// This forces the step() always use the fully connected matrix.
    ///
    /// The shape has to be [state_height*state_width, state_height*state_width].
    void reservoir_weights_full(af::array new_weights)
    {
        assert(new_weights.type() == DType);
        assert(new_weights.numdims() == 2);
        assert(new_weights.dims(0) == state_.elements());
        assert(new_weights.dims(1) == state_.elements());
        reservoir_w_full_ = std::move(new_weights);
        force_matmul_ = true;
    }

    /// Get the reservoir weights of the network.
    const af::array& reservoir_weights() const
    {
        return reservoir_w_;
    }

    /// Get the reservoir weights of the network in the form of fully connected matrix.
    const af::array& reservoir_weights_full() const
    {
        return reservoir_w_full_;
    }

    /// Set the reservoir biases of the network.
    ///
    /// The shape has to be the same as the state.
    void reservoir_biases(af::array new_biases)
    {
        assert(new_biases.type() == DType);
        assert(new_biases.dims() == state_.dims());
        reservoir_b_ = std::move(new_biases);
    }

    /// Get the reservoir biases of the network.
    const af::array& reservoir_biases() const
    {
        return reservoir_b_;
    }

    // Learning is not available at the moment.
    // /// Get the learning rate.
    // double learning_rate() const
    // {
    //     return learning_rate_;
    // }

    /// Set the learning rate.
    void learning_rate(double learning_rate) override
    {
        // Learning is not available at the moment.
        // learning_rate_ = learning_rate;
    }

    /// Disable random noise e.g., for lyapunov testing.
    void random_noise(bool enable) override
    {
        noise_enabled_ = enable;
    }

    std::unique_ptr<net_base> clone() const override
    {
        return std::make_unique<lcnn>(*this);
    }
};

/// Generate a random locally connected echo state network.
///
/// \param n_ins The number of inputs.
/// \param n_outs The number of outputs.
/// \param args The parameters by which is the network constructed.
template <af::dtype DType = af::dtype::f64>
lcnn<DType> random_lcnn(long n_ins, long n_outs, const po::variables_map& args, std::mt19937& prng)
{
    // The number of rows of the state matrix.
    long state_height = args.at("lcnn.state-height").as<long>();
    // The number of columns of the state matrix.
    long state_width = args.at("lcnn.state-width").as<long>();
    // The number of rows of the neuron kernel.
    long kernel_height = args.at("lcnn.kernel-height").as<long>();
    // The number of columns of the neuron kernel.
    long kernel_width = args.at("lcnn.kernel-width").as<long>();
    // Standard deviation of the normal distribution generating the reservoir.
    double sigma_res = args.at("lcnn.sigma-res").as<double>();
    // The mean of the normal distribution generating the reservoir.
    double mu_res = args.at("lcnn.mu-res").as<double>();
    // The input weight.
    double in_weight = args.at("lcnn.in-weight").as<double>();
    // The feedback weights will be generated from [0, fb_weight].
    double fb_weight = args.at("lcnn.fb-weight").as<double>();
    // Standard deviation of the normal distribution generating the biases.
    double sigma_b = args.at("lcnn.sigma-b").as<double>();
    // The mean of the normal distribution generating the biases.
    double mu_b = args.at("lcnn.mu-b").as<double>();
    // The sparsity of the reservoir weight matrix. For 0, the matrix is
    // fully connected. For 1, the matrix is completely zero.
    double sparsity = args.at("lcnn.sparsity").as<double>();
    // The reservoir topology.
    std::string topology = args.at("lcnn.topology").as<std::string>();
    // Put input to all neurons. In such a case, the input weights are
    // distributed uniformly from [0, in_weight].
    bool input_to_all = args.at("lcnn.input-to-all").as<bool>();

    if (kernel_height % 2 == 0 || kernel_width % 2 == 0)
        throw std::invalid_argument{"Kernel size has to be odd."};

    lcnn_config cfg{args};
    af::randomEngine af_prng{AF_RANDOM_ENGINE_DEFAULT, prng()};
    int neurons = state_height * state_width;
    // generate the reservoir weights based on topology
    if (topology == "sparse") {
        cfg.reservoir_w_full = sigma_res * af::randn({neurons, neurons}, DType, af_prng) + mu_res;
        // make the reservoir sparse by the given coefficient
        cfg.reservoir_w_full *=
          af::randu({cfg.reservoir_w_full.dims()}, DType, af_prng) >= sparsity;
    } else if (topology.starts_with("conv")) {
        // generate kernel
        af::array kernel =
          sigma_res * af::randn({kernel_height, kernel_width}, DType, af_prng) + mu_res;
        if (topology == "conv-od") {
            kernel(af::span, af::seq(kernel_width / 2, af::end)) = 0.;
        }
        // generate reservoir weights
        cfg.reservoir_w = af::tile(kernel, state_height * state_width);
        cfg.reservoir_w =
          af::moddims(cfg.reservoir_w, {state_height, state_width, kernel_height, kernel_width});
        // make the reservoir sparse by the given coefficient
        cfg.reservoir_w *= af::randu({cfg.reservoir_w.dims()}, DType, af_prng) >= sparsity;
    } else if (topology.starts_with("lcnn")) {
        // generate reservoir weights
        cfg.reservoir_w = sigma_res
            * af::randn({state_height, state_width, kernel_height, kernel_width}, DType, af_prng)
          + mu_res;
        // make the reservoir sparse by the given coefficient
        cfg.reservoir_w *= af::randu({cfg.reservoir_w.dims()}, DType, af_prng) >= sparsity;
        if (topology == "lcnn-noself") {
            cfg.reservoir_w(
              af::span, af::span, cfg.reservoir_w.dims(2) / 2, cfg.reservoir_w.dims(3) / 2) = 0.;
        } else if (topology == "lcnn-od") {
            // only allow connections going to the right
            cfg.reservoir_w(
              af::span, af::span, af::span, af::seq(cfg.reservoir_w.dims(3) / 2, af::end)) = 0.;
        } else if (topology == "lcnn-a1") {
            cfg.reservoir_w(
              af::span, af::span, af::span, af::seq(cfg.reservoir_w.dims(3) / 2, af::end)) = 0.;
            cfg.reservoir_w(
              af::span, af::span, af::seq(0, cfg.reservoir_w.dims(2) / 2 - 1), af::span) = 0.;
            cfg.reservoir_w(
              af::span, af::span, af::seq(cfg.reservoir_w.dims(2) / 2 + 1, af::end), af::span) = 0.;
            [[maybe_unused]] int kernel_expect_nonzero = cfg.reservoir_w.dims(3) / 2;
            assert(
              af::count<int>(cfg.reservoir_w)
              <= kernel_expect_nonzero * cfg.reservoir_w.dims(0) * cfg.reservoir_w.dims(1));
        } else if (topology == "lcnn-a3") {
            cfg.reservoir_w(
              af::span, af::span, af::span, af::seq(cfg.reservoir_w.dims(3) / 2 + 1, af::end)) = 0.;
            cfg.reservoir_w(
              af::span, af::span, af::seq(0, cfg.reservoir_w.dims(2) / 2),
              af::seq(cfg.reservoir_w.dims(3) / 2, af::end)) = 0.;
            [[maybe_unused]] int kernel_expect_nonzero =
              cfg.reservoir_w.dims(2) / 2 + cfg.reservoir_w.dims(2) * (cfg.reservoir_w.dims(3) / 2);
            assert(
              af::count<int>(cfg.reservoir_w)
              <= kernel_expect_nonzero * cfg.reservoir_w.dims(0) * cfg.reservoir_w.dims(1));
        } else if (topology == "lcnn-a4") {
            cfg.reservoir_w(
              af::span, af::span, af::span, af::seq(cfg.reservoir_w.dims(3) / 2 + 1, af::end)) = 0.;
            cfg.reservoir_w(
              af::span, af::span, af::seq(0, cfg.reservoir_w.dims(2) / 2 - 1), af::span) = 0.;
            // center
            cfg.reservoir_w(
              af::span, af::span, cfg.reservoir_w.dims(2) / 2, cfg.reservoir_w.dims(3) / 2) = 0.;
            // bottom left
            cfg.reservoir_w(
              af::span, af::span, af::seq(cfg.reservoir_w.dims(2) / 2 + 1, af::end),
              af::seq(0, cfg.reservoir_w.dims(3) / 2 - 1)) = 0.;
            [[maybe_unused]] int kernel_expect_nonzero =
              cfg.reservoir_w.dims(2) / 2 + cfg.reservoir_w.dims(3) / 2;
            assert(
              af::count<int>(cfg.reservoir_w)
              <= kernel_expect_nonzero * cfg.reservoir_w.dims(0) * cfg.reservoir_w.dims(1));
        } else if (topology == "lcnn-a5") {
            cfg.reservoir_w(
              af::span, af::span, af::span, af::seq(cfg.reservoir_w.dims(3) / 2, af::end)) = 0.;
            cfg.reservoir_w(
              af::span, af::span, af::seq(0, cfg.reservoir_w.dims(2) / 2 - 1), af::span) = 0.;
            [[maybe_unused]] int kernel_expect_nonzero =
              (cfg.reservoir_w.dims(2) / 2 + 1) * (cfg.reservoir_w.dims(3) / 2);
            assert(
              af::count<int>(cfg.reservoir_w)
              <= kernel_expect_nonzero * cfg.reservoir_w.dims(0) * cfg.reservoir_w.dims(1));
        }
    } else if (topology.starts_with("const")) {
        // TODO what if we disable only self-connections?
        double c = std::normal_distribution{mu_res, sigma_res}(prng);
        // generate reservoir weights
        cfg.reservoir_w =
          af::constant(c, {state_height, state_width, kernel_height, kernel_width}, DType);
        // make the reservoir sparse by the given coefficient
        cfg.reservoir_w *= af::randu({cfg.reservoir_w.dims()}, DType, af_prng) >= sparsity;
        // only allow connections going to the right
        if (topology == "const-od") {
            cfg.reservoir_w(
              af::span, af::span, af::span, af::seq(cfg.reservoir_w.dims(3) / 2, af::end)) = 0.;
        } else if (topology == "const-lindiscount") {
            af::array mask;
            if (kernel_width == 3)
                mask = af::constant(1, 1);
            else
                mask = af::transpose(1. / af::seq(kernel_width / 2, 1, -1));
            mask = af::join(1, mask, af::constant(0, 1), af::flip(mask, 1));
            mask = af::tile(mask, state_height * state_width * kernel_height);
            mask = af::moddims(mask, state_height, state_width, kernel_height, kernel_width);
            cfg.reservoir_w *= mask;
        } else if (topology == "const-od-lindiscount") {
            cfg.reservoir_w(
              af::span, af::span, af::span, af::seq(cfg.reservoir_w.dims(3) / 2, af::end)) = 0.;
            af::array mask;
            if (kernel_width == 3)
                mask = af::constant(1, 1);
            else
                mask = af::transpose(1. / af::seq(kernel_width / 2, 1, -1));
            mask = af::tile(mask, state_height * state_width * kernel_height);
            mask = af::moddims(mask, state_height, state_width, kernel_height, kernel_width / 2);
            cfg.reservoir_w(
              af::span, af::span, af::span, af::seq(0, cfg.reservoir_w.dims(3) / 2 - 1)) *= mask;
        }
    } else if (topology == "permutation") {
        // only allow one connection to each neuron
        std::vector<int> perm(neurons, 0);
        std::iota(perm.begin(), perm.end(), 0);
        std::shuffle(perm.begin(), perm.end(), prng);
        // build an eye matrix with permuted columns
        cfg.reservoir_w_full = af::constant(0, neurons, neurons, DType);
        for (int i = 0; i < neurons; ++i) cfg.reservoir_w_full(i, perm.at(i)) = 1;
        // Assign random weights to the permuted eye matrix.
        cfg.reservoir_w_full = cfg.reservoir_w_full * sigma_res
            * af::randn({cfg.reservoir_w_full.dims()}, DType, af_prng)
          + cfg.reservoir_w_full * mu_res;
        // make the reservoir sparse by the given coefficient
        cfg.reservoir_w_full *=
          af::randu({cfg.reservoir_w_full.dims()}, DType, af_prng) >= sparsity;
    } else if (topology == "ring" || topology == "chain") {
        cfg.reservoir_w_full = af::constant(0, neurons, neurons, DType);
        for (int i = 0; i < neurons; ++i) cfg.reservoir_w_full(i, (i + 1) % neurons) = 1;
        if (topology == "chain") cfg.reservoir_w_full(neurons - 1, 0) = 0;
        // Assign random weights to the ring matrix.
        cfg.reservoir_w_full = cfg.reservoir_w_full * sigma_res
            * af::randn({cfg.reservoir_w_full.dims()}, DType, af_prng)
          + cfg.reservoir_w_full * mu_res;
        // make the reservoir sparse by the given coefficient
        cfg.reservoir_w_full *=
          af::randu({cfg.reservoir_w_full.dims()}, DType, af_prng) >= sparsity;
    } else {
        throw std::runtime_error{"Unknown topology `" + topology + "`."};
    }

    // generate reservoir biases
    cfg.reservoir_b = sigma_b * af::randn({state_height, state_width}, DType, af_prng) + mu_b;

    // for improved visualizations, this is a list of nice positions
    // in the state matrix.
    const std::vector<std::pair<long, long>> nice_positions = {
      {state_height / 2, state_width / 2},  // center
      {state_height / 2, 0},                // left center
      {state_height - 1, state_width / 2},  // bottom center
      {state_height - 1, 0},                // bottom left corner
    };
    auto free_position = nice_positions.begin();
    // if we run out of nice positions, have a generator prepared
    std::uniform_int_distribution<long> vert_dist{0, state_height - 1};
    std::uniform_int_distribution<long> horiz_dist{0, state_width - 1};

    if (input_to_all) {
        // put input and feedback into all the neurons
        cfg.input_w = af::randu({state_height, state_width, n_ins}, DType, af_prng) * in_weight;
        cfg.feedback_w = af::randu({state_height, state_width, n_outs}, DType, af_prng) * fb_weight;
    } else {
        // choose the locations for inputs and feedbacks
        cfg.input_w = af::constant(0, state_height, state_width, n_ins, DType);
        for (long i = 0; i < n_ins; ++i) {
            if (free_position != nice_positions.end()) {
                cfg.input_w(free_position->first, free_position->second, i) = in_weight;
                ++free_position;
            } else {
                cfg.input_w(vert_dist(prng), horiz_dist(prng), i) = in_weight;
            }
        }
        cfg.feedback_w = af::constant(0, state_height, state_width, n_outs, DType);
        for (long i = 0; i < n_outs; ++i) {
            if (free_position != nice_positions.end()) {
                cfg.feedback_w(free_position->first, free_position->second, i) = fb_weight;
                ++free_position;
            } else {
                cfg.feedback_w(vert_dist(prng), horiz_dist(prng), i) = fb_weight;
            }
        }
    }

    // the initial state is full of zeros
    cfg.init_state = af::constant(0, state_height, state_width, DType);
    // the initial state is random
    // cfg.init_state = af::randu({state_height, state_width}, DType, af_prng) * 2. - 1.;

    return lcnn<DType>{std::move(cfg), prng};
}

/// Locally connected network options description for command line parsing.
po::options_description lcnn_arg_description()
{
    po::options_description lcnn_arg_desc{"Locally connected network options"};
    lcnn_arg_desc.add_options()                                    //
      ("lcnn.state-height", po::value<long>()->default_value(11),  //
       "The fixed height of the kernel.")                          //
      ("lcnn.state-width", po::value<long>()->default_value(11),   //
       "The width of the state matrix.")                           //
      ("lcnn.kernel-height", po::value<long>()->default_value(5),  //
       "The height of the kernel.")                                //
      ("lcnn.kernel-width", po::value<long>()->default_value(5),   //
       "The width of the kernel.")                                 //
      ("lcnn.sigma-res", po::value<double>()->default_value(0.2),  //
       "See random_lcnn().")                                       //
      ("lcnn.mu-res", po::value<double>()->default_value(0),       //
       "See random_lcnn().")                                       //
      ("lcnn.in-weight", po::value<double>()->default_value(0.1),  //
       "See random_lcnn().")                                       //
      ("lcnn.fb-weight", po::value<double>()->default_value(0),    //
       "See random_lcnn().")                                       //
      ("lcnn.sigma-b", po::value<double>()->default_value(0),      //
       "See random_lcnn().")                                       //
      ("lcnn.mu-b", po::value<double>()->default_value(0),         //
       "See random_lcnn().")                                       //
      ("lcnn.sparsity", po::value<double>()->default_value(0),     //
       "See random_lcnn().")                                       //

      // Learning is not available at the moment.
      // ("lcnn.learning-rate", po::value<double>()->default_value(0),
      //    "See lcnn_config struct.")
      // ("lcnn.learning-rate-ema", po::value<double>()->default_value(0),
      //    "See lcnn_config struct.")
      // ("lcnn.learning-rate-diff", po::value<double>()->default_value(0),
      //    "See lcnn_config struct.")
      // ("lcnn.learning-bias", po::value<double>()->default_value(0),
      //    "See lcnn_config struct.")
      // ("lcnn.learning-rate-bias", po::value<double>()->default_value(0),
      //    "See lcnn_config struct.")
      // ("lcnn.weight-decay", po::value<double>()->default_value(0),
      //    "See random_lcnn().")
      // ("lcnn.state-ema-decay", po::value<double>()->default_value(0),
      //    "See random_lcnn().")

      ("lcnn.topology", po::value<std::string>()->default_value("sparse"),  //
       "See random_lcnn().")                                                //
      ("lcnn.input-to-all", po::value<bool>()->default_value(false),        //
       "See random_lcnn().")                                                //
      ("lcnn.random-spike-prob", po::value<double>()->default_value(0),     //
       "See lcnn_config class.")                                            //
      ("lcnn.random-spike-std", po::value<double>()->default_value(0),      //
       "See lcnn_config class.")                                            //
      ("lcnn.noise", po::value<double>()->default_value(0),                 //
       "See lcnn_config class.")                                            //
      ("lcnn.leakage", po::value<double>()->default_value(1),               //
       "See lcnn_config class.")                                            //
      ;
    return lcnn_arg_desc;
}

std::unique_ptr<net_base>
make_net(long n_ins, long n_outs, const po::variables_map& args, std::mt19937& prng)
{
    if (args.at("gen.net-type").as<std::string>() == "lcnn") {
        return std::make_unique<lcnn<af::dtype::f64>>(random_lcnn(n_ins, n_outs, args, prng));
    }
    if (args.at("gen.net-type").as<std::string>() == "simple-esn") {
        return std::make_unique<simple_esn<af::dtype::f64>>(random_esn(n_ins, n_outs, args, prng));
    }
    throw std::runtime_error{
      "Unknown net type \"" + args.at("gen.net-type").as<std::string>() + "\"."};
}

}  // namespace esn
