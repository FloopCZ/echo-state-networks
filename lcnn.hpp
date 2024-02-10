#pragma once

// Locally connected Echo state network class and training functions. //

#include "arrayfire_utils.hpp"
#include "common.hpp"
#include "data_map.hpp"
#include "lcnn_step.hpp"
#include "net.hpp"
#include "simple_esn.hpp"

#include <arrayfire.h>
#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>
#include <cassert>
#include <cmath>
#include <fmt/format.h>
#include <limits>
#include <random>
#include <range/v3/all.hpp>
#include <stdexcept>

namespace esn {

namespace po = boost::program_options;
namespace rgv = ranges::views;
namespace rga = ranges::actions;

/// Locally connected network configuration.
struct lcnn_config {
    /// Input and output names.
    std::set<std::string> input_names;
    std::set<std::string> output_names;
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
    /// The L2 regularization coefficient.
    double l2 = 0;
    /// The L2 regularization coefficient.
    long intermediate_steps = 1;
    /// The number of training trials (select random indices, train, repeat).
    long n_train_trials = 1;
    /// Indices of neurons used as predictors during training.
    /// Leave empty to use all neurons.
    long n_state_predictors = 0;
    // How should the result of multiple calls to train() be aggregated.
    std::string train_aggregation = "ensemble";
    // The probability that a single data point belongs to the valid set during train trial.
    double valid_train_ratio = 0.2;

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
        l2 = args.at("lcnn.l2").as<double>();
        intermediate_steps = args.at("lcnn.intermediate-steps").as<long>();
        n_train_trials = args.at("lcnn.n-train-trials").as<long>();
        n_state_predictors = args.at("lcnn.n-state-predictors").as<long>();
        train_aggregation = args.at("lcnn.train-aggregation").as<std::string>();
        valid_train_ratio = args.at("lcnn.valid-train-ratio").as<double>();
    }
};

/// Advanced Echo State Networks with various reservoir topologies.
template <af::dtype DType = DEFAULT_AF_DTYPE>
class lcnn : public net_base {
protected:
    struct indiced_output_w {
        af::array state_predictor_indices;
        af::array output_w;
    };

    std::set<std::string> input_names_;
    std::set<std::string> output_names_;
    af::array state_delta_;  // working variable used during the step function
    af::array state_;
    data_map last_output_;  // the last output of the net as a data map
    data_map prev_step_feedback_;
    af::array reservoir_w_;
    af::array reservoir_w_full_;
    af::array reservoir_b_;
    af::array input_w_;
    af::array feedback_w_;
    std::vector<indiced_output_w> indiced_output_w_;
    bool force_matmul_;

    // Random engines.
    std::mt19937 prng_init_;
    std::mt19937 prng_;
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
    double l2_;
    long intermediate_steps_;
    long n_train_trials_;
    long n_state_predictors_;
    std::string train_aggregation_;
    double valid_train_ratio_;

    /// Return whether the step should be performed by matmul or by the lcnn step function.
    bool do_matmul_step() const
    {
        if (force_matmul_) return true;
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
        return lcnn_step(state, reservoir_w_);
    }

    /// Update the state matrix using the lcnn step function.
    virtual void update_via_weights()
    {
        assert(!force_matmul_);
        state_delta_ = update_via_weights_impl(state_);
    }

    /// Update the state matrix by adding the inputs.
    virtual void update_via_input(const af::array& input)
    {
        af::array input_w =
          af::moddims(input_w_, state_.dims(0) * state_.dims(1), input_names_.size());
        af::array delta = af::matmul(std::move(input_w), input);
        state_delta_ += af::moddims(std::move(delta), state_.dims());
    }

    /// Update the state matrix by adding the feedback.
    virtual void update_via_feedback(const af::array& feedback)
    {
        af::array feedback_w =
          af::moddims(feedback_w_, state_.dims(0) * state_.dims(1), output_names_.size());
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
    virtual void update_last_output()
    {
        if (indiced_output_w_.empty()) {
            last_output_.clear();
            return;
        }
        af::array output =
          af::constant(af::NaN, output_names_.size(), indiced_output_w_.size(), state_.type());
        // Evaluate every output_w and aggregate them to the final output.
        for (int i = 0; i < indiced_output_w_.size(); ++i) {
            const indiced_output_w& iow = indiced_output_w_.at(i);
            af::array predictors = af::flat(state_);
            if (!iow.state_predictor_indices.isempty())
                predictors = predictors(iow.state_predictor_indices);
            predictors = af_utils::add_ones(predictors, 0);
            output(af::span, i) = af::matmul(iow.output_w, predictors);
        }
        assert(train_aggregation_ == "ensemble" || train_aggregation_ == "delta");
        if (train_aggregation_ == "ensemble") output = af::median(output, 1);
        if (train_aggregation_ == "delta") output = af::sum(output, 1);
        last_output_ = {output_names_, output};
        assert(last_output_.data().dims() == (af::dim4{output_names_.size()}));
        assert(af::allTrue<bool>(!af::isNaN(af::flat(last_output_.data()))));
    }

    virtual void update_last_output_via_teacher_force(const data_map& step_feedback)
    {
        // consider NaN as not provided
        data_map nonnan_step_feedback = step_feedback.drop_nan();
        if (nonnan_step_feedback.empty()) return;
        // check that with empty last output, full feedback is provided.
        assert(!last_output_.empty() || nonnan_step_feedback.keys() == output_names_);
        last_output_ = nonnan_step_feedback.extend(last_output_).filter(output_names_);
    }

    /// Generate random indices to a flattened state matrix.
    af::array generate_random_state_indices(long n)
    {
        af::array idxs = af_utils::shuffle(af::seq(state_.elements()), af_prng_)(af::seq(n));
        return af::sort(std::move(idxs));
    }

public:
    lcnn() = default;

    /// Locally connected echo state network constructor.
    lcnn(lcnn_config cfg, std::mt19937 prng)
      : input_names_{cfg.input_names}
      , output_names_{cfg.output_names}
      , last_output_{}
      , force_matmul_{false}
      , prng_init_{std::move(prng)}
      , prng_{prng_init_}
      , af_prng_{AF_RANDOM_ENGINE_DEFAULT, prng_()}

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
      , l2_{cfg.l2}
      , intermediate_steps_{cfg.intermediate_steps}
      , n_train_trials_{cfg.n_train_trials}
      , n_state_predictors_{cfg.n_state_predictors}
      , train_aggregation_{cfg.train_aggregation}
      , valid_train_ratio_{cfg.valid_train_ratio}
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

    /// TODO fix docs
    /// Perform a single step with a single input.
    /// \param input The input value.
    /// \param feedback The teacher-forced feedback to be used instead
    ///                 of the network's output.
    /// \param desired The desired output. This is only used for callbacks.
    ///                Has to be of size [n_outs].
    void step(
      const data_map& step_input,
      const data_map& step_feedback,
      const data_map& step_desired,
      input_transform_fn_t input_transform) override
    {
        // TODO desired and feedback is the same, only one should be provided and there should
        // be teacher-force bool param
        update_last_output_via_teacher_force(prev_step_feedback_);
        prev_step_feedback_ = step_feedback;

        // prepare the inputs for this step
        data_map tr_last_output = input_transform(last_output_);
        data_map tr_step_input =
          input_transform(step_input.drop_nan()).extend(tr_last_output).filter(input_names_);

        // validate all input data
        assert(tr_step_input.length() == 1);
        assert(tr_step_input.keys() == input_names_);
        assert(af::allTrue<bool>(tr_step_input.data() >= -1. && tr_step_input.data() <= 1.));
        if (!step_feedback.empty()) {
            assert(step_feedback.length() == 1);
        }
        if (!step_desired.empty()) {
            assert(step_desired.length() == 1);
            assert(step_desired.keys() == output_names_);
        }

        // Update the internal state.
        for (long interm_step = 0; interm_step < intermediate_steps_; ++interm_step) {
            // Perform matrix multiplication instead of state unwrapping for large kernels.
            if (do_matmul_step()) {
                update_via_weights_matmul();
            } else {
                // Use state unwrapping for small kernels
                update_via_weights();
            }

            // add input
            update_via_input(tr_step_input.data());

            // add feedback
            if (!tr_last_output.empty()) {
                assert(tr_last_output.keys() == output_names_);
                update_via_feedback(tr_last_output.data());
            }

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
            assert(!af::anyTrue<bool>(af::isNaN(state_)));
        }

        update_last_output();

        // Call the registered callback functions.
        for (on_state_change_callback_t& fnc : on_state_change_callbacks_) {
            on_state_change_data data = {
              .state = state_,
              .input = {.input = step_input, .feedback = step_feedback, .desired = step_desired},
              .output = last_output_,
              .event = event_};
            fnc(*this, std::move(data));
        }
        event_ = std::nullopt;
    }

    const data_map& last_output() const
    {
        return last_output_;
    }

    void last_output(data_map value)
    {
        last_output_ = std::move(value);
    }

    /// TODO fix docs
    /// Perform multiple steps with multiple input seqences.
    /// \param inputs Input sequence of dimensions [n_ins, time].
    /// \param feedback The desired output sequences to be teacher-forced into the net.
    ///                 Needs to have dimensions [n_outs, time]
    /// \param desired The desired output. This is only used for callbacks.
    ///                Has to be of size [n_outs, time].
    /// \return The array of intermediate states of dimensions [state_height, state_width, time]
    ///         and the array of intermediate outputs of dimensions [n_outs, time].
    feed_result_t feed(const input_t& input) override
    {
        long data_len = -1;
        auto check_data = [&data_len](const data_map& dm) {
            if (dm.empty()) return;
            assert(dm.data().type() == DType);
            assert(dm.data().numdims() <= 2);
            assert(dm.size() > 0);
            assert(data_len == -1 || dm.length() == data_len);
            assert(!af::anyTrue<bool>(af::isInf(dm.data())));
            data_len = dm.length();
        };
        check_data(input.input);
        check_data(input.feedback);
        check_data(input.desired);

        feed_result_t result;
        result.states = af::array(state_.dims(0), state_.dims(1), data_len, DType);
        result.outputs = af::constant(af::NaN, output_names_.size(), data_len, DType);
        result.desired = input.desired.data();
        for (long i = 0; i < data_len; ++i) {
            // prepare the inputs for this step
            data_map step_input = input.input.select(i);
            data_map step_feedback = input.feedback.select(i);
            data_map step_desired = input.desired.select(i);
            step(step_input, step_feedback, step_desired, input.input_transform);
            result.states(af::span, af::span, i) = state_;
            if (!last_output_.empty()) result.outputs(af::span, i) = last_output_.data();
        }
        return result;
    }

    // TODO fix docs
    /// Train the network on the given sequence.
    /// \param input Input sequence of dimensions [n_ins, time].
    /// \param desired The desired output sequences. Those are also teacher-forced into the net.
    ///                Needs to have dimensions [n_outs, time]
    train_result_t train(const input_t& input) override
    {
        return train(feed(input));
    }

    /// Clear the network output weights and reset prng to the initial state.
    void reset() override
    {
        prng_ = prng_init_;
        af_prng_ = af::randomEngine{AF_RANDOM_ENGINE_DEFAULT, prng_()};
        indiced_output_w_.clear();
    }

    /// Train the network on already processed feed result.
    /// \param data Training data.
    train_result_t
    train_impl(const feed_result_t& data, const af::array& state_predictor_indices) const
    {
        assert(data.states.type() == DType);
        assert(
          (data.states.dims() == af::dim4{state_.dims(0), state_.dims(1), data.outputs.dims(1)}));
        assert(data.outputs.type() == DType);
        assert(data.outputs.numdims() <= 2);
        assert(data.outputs.dims(0) == output_names_.size());
        if (!data.desired) throw std::runtime_error{"No desired data to train to."};
        assert(data.outputs.dims(1) == data.desired->dims(1));
        assert(data.desired->type() == DType);
        assert(data.desired->numdims() <= 2);
        assert(data.desired->dims(0) == output_names_.size());
        assert(!af::anyTrue<bool>(af::isNaN(data.states)));
        assert(!af::anyTrue<bool>(af::isNaN(*data.desired)));
        af::array predictors = af::moddims(data.states, state_.elements(), data.desired->dims(1));
        if (!state_predictor_indices.isempty())
            predictors = predictors(state_predictor_indices, af::span);
        predictors = af_utils::add_ones(predictors.T(), 1);
        af::array output_w = af_utils::solve(predictors, data.desired->T(), l2_).T();
        output_w(af::isNaN(output_w) || af::isInf(output_w)) = 0.;
        assert(output_w.dims() == (af::dim4{output_names_.size(), predictors.dims(1)}));
        return {.predictors = std::move(predictors), .output_w = std::move(output_w)};
    }

    /// Train the network on already processed feed result.
    /// \param data Training data.
    train_result_t train(const feed_result_t data) override
    {
        if (!data.desired) throw std::runtime_error{"No desired data to train to."};

        feed_result_t train_trial_data = data;
        if (!indiced_output_w_.empty()) {
            double feed_err = af_utils::mse<double>(data.outputs, *data.desired);
            std::cout << fmt::format(
              "Before train {} MSE error: {}", indiced_output_w_.size(), feed_err)
                      << std::endl;
        }

        assert(train_aggregation_ == "ensemble" || train_aggregation_ == "delta");
        // Prepare for delta training (epoch 1 and later).
        // In the second and later epochs, we only train the difference.
        if (train_aggregation_ == "delta" && !indiced_output_w_.empty())
            train_trial_data.desired = *data.desired - data.outputs;

        struct train_trial_result_t {
            train_result_t train_result;
            af::array state_predictor_indices;
            double valid_err;
        } best_train_result{.valid_err = std::numeric_limits<double>::max()};

        bool predictor_subset = n_state_predictors_ > 0 && n_state_predictors_ < state_.elements();
        bool cross_validate = predictor_subset && n_train_trials_ > 1;

        // split the data to train/valid if not using all state neurons or there is just a single
        // train trial
        feed_result_t train_data = train_trial_data;
        feed_result_t valid_data = train_trial_data;
        if (cross_validate) {
            af::array train_set_idx =
              af::randu(train_trial_data.states.dims(2), af::dtype::f32, af_prng_)
              < valid_train_ratio_;
            af::array valid_set_idx = !train_set_idx;
            train_data = {
              .states = train_trial_data.states(af::span, af::span, train_set_idx),
              .outputs = train_trial_data.outputs(af::span, train_set_idx),
              .desired = (*train_trial_data.desired)(af::span, train_set_idx)};
            valid_data = {
              .states = train_trial_data.states(af::span, af::span, valid_set_idx),
              .outputs = train_trial_data.outputs(af::span, valid_set_idx),
              .desired = (*train_trial_data.desired)(af::span, valid_set_idx)};
        }

        assert(n_train_trials_ > 0);
        for (long i = 0; i < n_train_trials_; ++i) {
            // select random state predictor indices
            af::array state_predictor_indices;
            if (predictor_subset)
                state_predictor_indices = generate_random_state_indices(n_state_predictors_);
            // train
            train_result_t train_result = train_impl(train_data, state_predictor_indices);
            af::array train_prediction =
              af::matmulNT(train_result.predictors, train_result.output_w);
            double train_err = af_utils::mse<double>(train_prediction.T(), *train_data.desired);
            double valid_err = std::numeric_limits<double>::quiet_NaN();
            // predict out of sample
            if (cross_validate) {
                af::array valid_predictors =
                  af::moddims(valid_data.states, state_.elements(), valid_data.states.dims(2));
                if (!state_predictor_indices.isempty())
                    valid_predictors = valid_predictors(state_predictor_indices, af::span);
                valid_predictors = af_utils::add_ones(valid_predictors.T(), 1);
                af::array valid_prediction = af::matmulNT(valid_predictors, train_result.output_w);
                valid_err = af_utils::mse<double>(valid_prediction.T(), *valid_data.desired);
            }
            // print statistics
            std::cout << fmt::format(
              "Train {} trial {} MSE error (train): {}", indiced_output_w_.size(), i, train_err)
                      << std::endl;
            if (cross_validate)
                std::cout << fmt::format(
                  "Train {} trial {} MSE error (valid): {}", indiced_output_w_.size(), i, valid_err)
                          << std::endl;
            // select the best train trial
            if (state_predictor_indices.isempty() || valid_err < best_train_result.valid_err)
                best_train_result = {
                  .train_result = std::move(train_result),
                  .state_predictor_indices = state_predictor_indices,
                  .valid_err = valid_err};
            // no need to keep on trying if use all state predictors (the result will be the same)
            if (!cross_validate) break;
        }

        indiced_output_w_.push_back(
          {.state_predictor_indices = best_train_result.state_predictor_indices,
           .output_w = best_train_result.train_result.output_w});
        update_last_output();
        return best_train_result.train_result;
    }

    /// Clear the stored feedback which would otherwise be used in the next step.
    void clear_feedback() override
    {
        prev_step_feedback_.clear();
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
        // assert(new_state.numdims() == 2);  // Not true for vector state.
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

    /// The input names.
    const std::set<std::string>& input_names() const override
    {
        return input_names_;
    }

    /// The output names.
    const std::set<std::string>& output_names() const override
    {
        return output_names_;
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
        assert(
          (new_weights.dims() == af::dim4{state_.dims(0), state_.dims(1), input_names_.size()}));
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
        assert(
          (new_weights.dims() == af::dim4{state_.dims(0), state_.dims(1), output_names_.size()}));
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

        // Precalculate the fully connected weight matrix.
        if (do_matmul_step()) {
            int state_height = reservoir_w_.dims(0);
            int state_width = reservoir_w_.dims(1);
            int kernel_height = reservoir_w_.dims(2);
            int kernel_width = reservoir_w_.dims(3);

            // Convert the reservoir matrices on host for performance.
            std::vector<double> reservoir_w = af_utils::to_vector(reservoir_w_);
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
                            assert(full_index >= 0 && full_index < (long)reservoir_w_full.size());
                            int sparse_index = i + j * state_height + k * state_height * state_width
                              + l * state_height * state_width * kernel_height;
                            assert(sparse_index >= 0 && sparse_index < (long)reservoir_w.size());
                            reservoir_w_full[full_index] += reservoir_w[sparse_index];
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
            assert(af_utils::almost_equal(state_matmul, state_wrap, 1e-8));
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
template <af::dtype DType = DEFAULT_AF_DTYPE>
lcnn<DType> random_lcnn(
  const std::set<std::string>& input_names,
  const std::set<std::string>& output_names,
  const po::variables_map& args,
  std::mt19937& prng)
{
    long n_ins = input_names.size();
    long n_outs = output_names.size();
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
    // The input weight for each input.
    std::vector<double> in_weight = args.at("lcnn.in-weight").as<std::vector<double>>();
    if (in_weight.size() < n_ins)
        throw std::invalid_argument{
          fmt::format("in-weight ({}) < n_ins ({})", in_weight.size(), n_ins)};
    // The feedback weights will be generated from [0, fb_weight].
    std::vector<double> fb_weight = args.at("lcnn.fb-weight").as<std::vector<double>>();
    if (fb_weight.size() < n_outs)
        throw std::invalid_argument{
          fmt::format("fb-weight ({}) < n_outs ({})", fb_weight.size(), n_outs)};
    // Standard deviation of the normal distribution generating the biases.
    double sigma_b = args.at("lcnn.sigma-b").as<double>();
    // The mean of the normal distribution generating the biases.
    double mu_b = args.at("lcnn.mu-b").as<double>();
    // The sparsity of the reservoir weight matrix. For 0, the matrix is
    // fully connected. For 1, the matrix is completely zero.
    double sparsity = args.at("lcnn.sparsity").as<double>();
    // The reservoir topology.
    std::string topology = args.at("lcnn.topology").as<std::string>();
    std::set<std::string> topo_params;
    boost::split(topo_params, topology, boost::is_any_of("-"));
    // How many neurons are injected with each input.
    long input_to_n = args.at("lcnn.input-to-n").as<long>();

    if (kernel_height % 2 == 0 || kernel_width % 2 == 0)
        throw std::invalid_argument{"Kernel size has to be odd."};

    lcnn_config cfg{args};
    cfg.input_names = input_names;
    cfg.output_names = output_names;
    af::randomEngine af_prng{AF_RANDOM_ENGINE_DEFAULT, prng()};
    int neurons = state_height * state_width;
    // generate the reservoir weights based on topology
    if (topo_params.contains("sparse")) {
        cfg.reservoir_w_full = sigma_res * af::randn({neurons, neurons}, DType, af_prng) + mu_res;
        // make the reservoir sparse by the given coefficient
        cfg.reservoir_w_full *=
          af::randu({cfg.reservoir_w_full.dims()}, DType, af_prng) >= sparsity;
    } else if (topo_params.contains("conv")) {
        // generate kernel
        af::array kernel =
          sigma_res * af::randn({kernel_height, kernel_width}, DType, af_prng) + mu_res;
        if (topo_params.contains("od")) {
            kernel(af::span, af::seq(kernel_width / 2, af::end)) = 0.;
        }
        // generate reservoir weights
        cfg.reservoir_w = af::tile(af::flat(kernel), state_height * state_width);
        cfg.reservoir_w =
          af::moddims(cfg.reservoir_w, {kernel_height, kernel_width, state_height, state_width});
        cfg.reservoir_w = af::reorder(cfg.reservoir_w, 2, 3, 0, 1);
        assert(af::allTrue<bool>(
          cfg.reservoir_w(0, 0) == cfg.reservoir_w(state_height - 1, state_width - 1)));
        // make the reservoir sparse by the given coefficient
        cfg.reservoir_w *= af::randu({cfg.reservoir_w.dims()}, DType, af_prng) >= sparsity;
    } else if (topo_params.contains("lcnn")) {
        // generate reservoir weights
        cfg.reservoir_w = sigma_res
            * af::randn({state_height, state_width, kernel_height, kernel_width}, DType, af_prng)
          + mu_res;
        // make the reservoir sparse by the given coefficient
        cfg.reservoir_w *= af::randu({cfg.reservoir_w.dims()}, DType, af_prng) >= sparsity;
        if (topo_params.contains("noself")) {
            cfg.reservoir_w(
              af::span, af::span, cfg.reservoir_w.dims(2) / 2, cfg.reservoir_w.dims(3) / 2) = 0.;
        }
        if (topo_params.contains("od")) {
            // only allow connections going to the right
            cfg.reservoir_w(
              af::span, af::span, af::span, af::seq(cfg.reservoir_w.dims(3) / 2, af::end)) = 0.;
        }
        if (topo_params.contains("a1")) {
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
        }
        if (topo_params.contains("a3")) {
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
        }
        if (topo_params.contains("a4")) {
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
        }
        if (topo_params.contains("a5")) {
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
    } else if (topo_params.contains("const")) {
        // TODO what if we disable only self-connections?
        double c = std::normal_distribution{mu_res, sigma_res}(prng);
        // generate reservoir weights
        cfg.reservoir_w =
          af::constant(c, {state_height, state_width, kernel_height, kernel_width}, DType);
        // make the reservoir sparse by the given coefficient
        cfg.reservoir_w *= af::randu({cfg.reservoir_w.dims()}, DType, af_prng) >= sparsity;
        // only allow connections going to the right
        if (topo_params.contains("od")) {
            cfg.reservoir_w(
              af::span, af::span, af::span, af::seq(cfg.reservoir_w.dims(3) / 2, af::end)) = 0.;
        }
        if (topo_params.contains("lindiscount")) {
            af::array mask;
            if (kernel_width == 3)
                mask = af::constant(1, 1);
            else
                mask = af::transpose(1. / af::seq(kernel_width / 2, 1, -1));
            mask = af::join(1, mask, af::constant(0, 1), af::flip(mask, 1));
            mask = af::tile(mask, state_height * state_width * kernel_height);
            mask = af::moddims(mask, state_height, state_width, kernel_height, kernel_width);
            cfg.reservoir_w *= mask;
        }
    } else if (topo_params.contains("permutation")) {
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
    } else if (topo_params.contains("ring") || topo_params.contains("chain")) {
        cfg.reservoir_w_full = af::constant(0, neurons, neurons, DType);
        for (int i = 0; i < neurons; ++i) cfg.reservoir_w_full(i, (i + 1) % neurons) = 1;
        if (topo_params.contains("chain")) cfg.reservoir_w_full(neurons - 1, 0) = 0;
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

    if (topo_params.contains("nowrap") && !cfg.reservoir_w.isempty()) {
        for (int i = 0; i < kernel_width / 2; ++i) {
            cfg.reservoir_w(af::span, i, af::span, af::seq(0, kernel_width / 2 - i - 1)) = 0;
            cfg.reservoir_w(
              af::span, state_width - i - 1, af::span, af::seq(kernel_width / 2 + i + 1, af::end)) =
              0;
        }
        for (int i = 0; i < kernel_height / 2; ++i) {
            cfg.reservoir_w(i, af::span, af::seq(0, kernel_height / 2 - i - 1), af::span) = 0;
            cfg.reservoir_w(
              state_height - i - 1, af::span, af::seq(kernel_height / 2 + i + 1, af::end),
              af::span) = 0;
        }
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

    if (input_to_n == 0 || input_to_n == state_height * state_width) {
        // put input and feedback into all the neurons
        cfg.input_w = af::randu({state_height, state_width, n_ins}, DType, af_prng);
        for (long i = 0; i < n_ins; ++i) cfg.input_w(af::span, af::span, i) *= in_weight.at(i);
        cfg.feedback_w = af::randu({state_height, state_width, n_outs}, DType, af_prng);
        for (long i = 0; i < n_outs; ++i) cfg.feedback_w(af::span, af::span, i) *= fb_weight.at(i);
    } else {
        // choose the locations for inputs and feedbacks
        cfg.input_w = af::constant(0, state_height, state_width, n_ins, DType);
        for (long i = 0; i < n_ins; ++i) {
            if (input_to_n == 1 && free_position != nice_positions.end()) {
                cfg.input_w(free_position->first, free_position->second, i) = in_weight.at(i);
                ++free_position;
            } else {
                af::array input_w_single = af::constant(0, state_height, state_width, DType);
                af::array idxs = af_utils::shuffle(af::seq(state_height * state_width), af_prng)(
                  af::seq(input_to_n));
                input_w_single(idxs) = in_weight.at(i);
                cfg.input_w(af::span, af::span, i) = input_w_single;
            }
        }
        cfg.feedback_w = af::constant(0, state_height, state_width, n_outs, DType);
        for (long i = 0; i < n_outs; ++i) {
            if (input_to_n == 1 && free_position != nice_positions.end()) {
                cfg.feedback_w(free_position->first, free_position->second, i) = fb_weight.at(i);
                ++free_position;
            } else {
                af::array feedback_w_single = af::constant(0, state_height, state_width, DType);
                af::array idxs = af_utils::shuffle(af::seq(state_height * state_width), af_prng)(
                  af::seq(input_to_n));
                feedback_w_single(idxs) = fb_weight.at(i);
                cfg.feedback_w(af::span, af::span, i) = feedback_w_single;
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
inline po::options_description lcnn_arg_description()
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
      ("lcnn.in-weight",                                           //
       po::value<std::vector<double>>()                            //
         ->multitoken()                                            //
         ->default_value(std::vector<double>{0.1}, "0.1"),         //
       "See random_lcnn().")                                       //
      ("lcnn.fb-weight",                                           //
       po::value<std::vector<double>>()                            //
         ->multitoken()                                            //
         ->default_value(std::vector<double>{0}, "0"),             //
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

      ("lcnn.topology", po::value<std::string>()->default_value("sparse"),             //
       "See random_lcnn().")                                                           //
      ("lcnn.input-to-n", po::value<long>()->default_value(0),                         //
       "See random_lcnn().")                                                           //
      ("lcnn.random-spike-prob", po::value<double>()->default_value(0),                //
       "See lcnn_config class.")                                                       //
      ("lcnn.random-spike-std", po::value<double>()->default_value(0),                 //
       "See lcnn_config class.")                                                       //
      ("lcnn.noise", po::value<double>()->default_value(0),                            //
       "See lcnn_config class.")                                                       //
      ("lcnn.leakage", po::value<double>()->default_value(1),                          //
       "See lcnn_config class.")                                                       //
      ("lcnn.intermediate-steps", po::value<long>()->default_value(1),                 //
       "See lcnn_config class.")                                                       //
      ("lcnn.l2", po::value<double>()->default_value(0),                               //
       "See lcnn_config class.")                                                       //
      ("lcnn.n-train-trials", po::value<long>()->default_value(1),                     //
       "See random_lcnn().")                                                           //
      ("lcnn.n-state-predictors", po::value<long>()->default_value(0),                 //
       "How many neurons are used for regression training.")                           //
      ("lcnn.train-aggregation", po::value<std::string>()->default_value("ensemble"),  //
       "See lcnn_config class.")                                                       //
      ("lcnn.valid-train-ratio", po::value<double>()->default_value(0.2),              //
       "See lcnn_config class.")                                                       //
      ;
    return lcnn_arg_desc;
}

}  // namespace esn
