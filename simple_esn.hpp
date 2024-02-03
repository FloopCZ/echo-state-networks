#pragma once

// Echo state network class and training functions. //

#include "arrayfire_utils.hpp"
#include "common.hpp"
#include "data_map.hpp"
#include "net.hpp"

#include <arrayfire.h>
#include <boost/program_options.hpp>
#include <cassert>

namespace esn {

namespace po = boost::program_options;

/// Echo state network.
template <af::dtype DType = DEFAULT_AF_DTYPE>
class simple_esn : public net_base {
    // the number of reservoir neurons
    long n_;
    // the input neuron names
    std::set<std::string> input_names_;
    // the output neuron names
    std::set<std::string> output_names_;
    // reservoir connections
    // a matrix of size n x n
    af::array reservoir_w_;
    // input weights
    // each row are the input weights for a single input neuron
    af::array input_w_;
    // output weights
    // each col are the output weights for a single neuron, the first col is the intercept
    af::array output_w_;
    // feedback weights
    af::array feedback_w_;
    // biases
    // a single vector of lenght `n`
    af::array biases_;
    // whether the random noise is enabled/disabled
    bool noise_enabled_;
    // standard deviation of the state noise
    double noise_;
    // leakage rate
    double leakage_;
    // the current state
    af::array state_;
    // the last output of the net
    af::array last_output_;

    // random engines
    std::mt19937 prng_init_;
    std::mt19937 prng_;
    af::randomEngine af_prng_;

public:
    simple_esn() = default;

    /// Echo state network constructor.
    simple_esn(
      long n,
      std::set<std::string> input_names,
      std::set<std::string> output_names,
      af::array reservoir_w,
      af::array input_w,
      af::array feedback_w,
      af::array biases,
      double noise,
      double leakage,
      std::mt19937 prng)
      : n_{n}
      , input_names_{std::move(input_names)}
      , output_names_{std::move(output_names)}
      , reservoir_w_{std::move(reservoir_w)}
      , input_w_{std::move(input_w)}
      , output_w_{af::constant(af::NaN, output_names_.size(), n_ + 1, DType)}
      , feedback_w_{std::move(feedback_w)}
      , biases_{std::move(biases)}
      , noise_enabled_{true}
      , noise_{noise}
      , leakage_{leakage}
      , state_{af::constant(0, n_, DType)}
      , last_output_{af::constant(0, output_names_.size(), DType)}
      , prng_init_{std::move(prng)}
      , prng_{prng_init_}
      , af_prng_{AF_RANDOM_ENGINE_DEFAULT, prng_()}
    {
        // check types
        assert((reservoir_w_.type() == DType));
        assert((input_w_.type() == DType));
        assert((feedback_w_.type() == DType));
        assert((biases_.type() == DType));
        assert((state_.type() == DType));

        // check dimensions
        assert((reservoir_w_.dims() == af::dim4{n_, n_}));
        assert((input_w_.dims() == af::dim4{n_, input_names_.size()}));
        assert((feedback_w_.dims() == af::dim4{n_, output_names_.size()}));
        assert((biases_.dims() == af::dim4{n_}));
        assert((state_.dims() == af::dim4{n_}));
    }

    /// Echo state network constructor with no biases.
    simple_esn(
      long n,
      std::set<std::string> input_names,
      std::set<std::string> output_names,
      af::array reservoir_w,
      af::array input_w,
      af::array feedback_w,
      double noise,
      double leakage,
      std::mt19937 prng)
      : simple_esn{
        n,
        std::move(input_names),
        std::move(output_names),
        std::move(reservoir_w),
        std::move(input_w),
        std::move(feedback_w),
        af::constant(0, n, DType),
        noise,
        leakage,
        prng}
    {
    }

    /// Perform a single step with multiple inputs.
    /// \param input The input values of size [n_ins].
    /// \param feedback The teacher-forced feedback to be used instead
    ///                 of the network's output of size [n_outs].
    /// \param desired The desired output. This is only used for callbacks.
    ///                Has to be of size [n_outs].
    void step(
      const data_map& step_input,
      const data_map& step_feedback,
      const data_map& step_desired,
      input_transform_fn_t input_transform) override
    {
        data_map orig_step_input = step_input;

        // prepare the inputs for this step (add missing keys from last output)
        assert(!last_output_.isempty());
        data_map last_output{output_names_, last_output_};
        data_map tr_last_output = input_transform(last_output);
        data_map tr_step_input =
          input_transform(step_input).extend(tr_last_output).filter(input_names_);

        // validate all input data
        assert(tr_step_input.keys() == input_names_);
        assert(tr_step_input.length() == 1);
        assert(af::allTrue<bool>(tr_step_input.data() >= -1. && tr_step_input.data() <= 1.));
        if (!step_feedback.empty()) {
            assert(step_feedback.length() == 1);
        }
        if (!step_desired.empty()) {
            assert(step_desired.length() == 1);
            assert(step_desired.keys() == output_names_);
        }

        af::array weighted_input = af::matmul(input_w_, tr_step_input.data());
        af::array feedback_activation = af::matmul(feedback_w_, tr_last_output.data());
        af::array internal_activation = af::matmul(reservoir_w_, state_);
        af::array noise =
          1. + af::randn({state_.dims()}, DType, af_prng_) * noise_ * noise_enabled_;
        af::array state_before_activation = std::move(weighted_input)
          + std::move(internal_activation) + std::move(feedback_activation) + biases_;
        state_ *= 1. - leakage_;
        state_ += af::tanh(state_before_activation * noise);
        af::eval(state_);
        if (step_feedback.empty()) {
            update_output();
        } else {
            assert(step_feedback.keys() == output_names_ && "Only full feedback is supported.");
            last_output_ = step_feedback.data();
        }
        assert(last_output_.dims() == (af::dim4{output_names_.size()}));

        // Call the registered callback functions.
        for (on_state_change_callback_t& fnc : on_state_change_callbacks_) {
            on_state_change_data data = {
              .state = state_,
              .input = {.input = step_input, .feedback = step_feedback, .desired = step_desired},
              .output = {output_names_, last_output_},
              .event = event_};
            fnc(*this, std::move(data));
        }
        event_ = std::nullopt;
    }

    /// Perform multiple steps with multiple input seqences.
    /// \param input Input sequence of dimensions [n_ins, time].
    /// \param feedback The desired output sequences to be teacher-forced into the net.
    ///                 Needs to have dimensions [n_outs, time]
    /// \param desired The desired output. This is only used for callbacks.
    ///                Has to be of size [n_outs, time].
    /// \return The array of intermediate states of dimensions [n, time] and the array
    ///         of intermediate outputs of dimensions [n_outs, time].
    feed_result_t feed(const input_t& input) override
    {
        long data_len = -1;
        auto check_data = [&data_len](const data_map& dm) {
            if (dm.empty()) return;
            assert(dm.data().type() == DType);
            assert(dm.data().numdims() <= 2);
            assert(dm.size() > 0);
            assert(data_len == -1 || dm.length() == data_len);
            data_len = dm.length();
        };
        check_data(input.input);
        check_data(input.feedback);
        check_data(input.desired);

        feed_result_t result;
        result.states = af::array(n_, data_len, DType);
        result.outputs = af::array(output_names_.size(), data_len, DType);
        result.desired = input.desired.data();
        for (long i = 0; i < data_len; ++i) {
            // prepare the inputs for this step
            data_map step_input = input.input.select(i);
            data_map step_feedback = input.feedback.select(i);
            data_map step_desired = input.desired.select(i);
            step(step_input, step_feedback, step_desired, input.input_transform);
            result.states(af::span, i) = state_;
            result.outputs(af::span, i) = last_output_;
        }
        return result;
    }

    /// TODO fix docs
    /// Train the network on the given sequence.
    /// \param inputs Input sequence of dimensions [n_ins, time].
    /// \param desired The desired output sequences. Those are also teacher-forced into the net.
    ///                Needs to have dimensions [n_outs, time]
    train_result_t train(const input_t& input) override
    {
        return train(feed(input));
    }

    /// Train the network on already processed feed data.
    /// \param data Training data.
    train_result_t train(feed_result_t data) override
    {
        assert(data.states.type() == DType);
        assert((data.states.dims() == af::dim4{state_.dims(0), data.outputs.dims(1)}));
        assert(data.outputs.type() == DType);
        assert(data.outputs.numdims() <= 2);
        assert(data.outputs.dims(0) == output_names_.size());
        if (!data.desired) throw std::runtime_error{"No desired data to train to."};
        assert(data.outputs.dims(1) == data.desired->dims(1));
        assert(data.desired->type() == DType);
        assert(data.desired->numdims() <= 2);
        assert(data.desired->dims(0) == output_names_.size());
        data.outputs = af::array{};  // free memory
        data.states = data.states.T();
        af::array predictors = af_utils::add_ones(data.states, 1);
        data.states = af::array{};  // free memory
        data.desired = data.desired->T();
        output_w_ = af::solve(predictors, *data.desired).T();
        data.desired = af::array{};  // free memory
        output_w_(af::isNaN(output_w_) || af::isInf(output_w_)) = 0.;
        assert(
          output_w_.dims()
          == (af::dim4{output_names_.size(), input_names_.size() + state_.elements() + 1}));
        return {.predictors = std::move(predictors), .output_w = output_w_};
    }

    /// Clear the stored feedback which would otherwise be used in the next step.
    void clear_feedback() override
    {
        update_output();
    }

    // Update the output using the current state.
    void update_output()
    {
        af::array predictors = af_utils::add_ones(state_, 0);
        last_output_ = af::matmul(output_w_, predictors);
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
        state_ = std::move(new_state);
    }

    /// Get the reservoir weights of the network.
    const af::array& reservoir_weights() const
    {
        return reservoir_w_;
    }

    /// Get the input weights of the network.
    const af::array& input_weights() const
    {
        return input_w_;
    }

    /// The number of neurons.
    long n() const
    {
        return n_;
    }

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
        af::array in = af::sum(reservoir_w_ != 0, 1);
        return af::mean<double>(reservoir_w_);
    }

    /// Set the learning rate.
    void learning_rate(double) override
    {
        // Not implemented.
    }

    /// Disable random noise e.g., for lyapunov testing.
    void random_noise(bool enable) override
    {
        noise_enabled_ = enable;
    }

    void reset() override
    {
        prng_ = prng_init_;
        output_w_ = af::constant(af::NaN, output_names_.size(), n_ + 1, DType);
    }

    std::unique_ptr<net_base> clone() const override
    {
        return std::make_unique<simple_esn>(*this);
    }
};

/// Generate a random echo state network.
///
/// \tparam DType The arrayfire data type.
/// \param n_ins The number of inputs.
/// \param n_outs The number of outputs.
/// \param args The parameters by which is the network constructed.
template <af::dtype DType = DEFAULT_AF_DTYPE>
simple_esn<DType> random_esn(
  const std::set<std::string>& input_names,
  const std::set<std::string>& output_names,
  const po::variables_map& args,
  std::mt19937& prng)
{
    long n_ins = input_names.size();
    long n_outs = output_names.size();
    // The number of neurons.
    long n = args.at("esn.neurons").as<long>();
    // Standard deviation of the normal distribution generating the reservoir.
    double sigma_res = args.at("esn.sigma-res").as<double>();
    // The mean of the normal distribution generating the reservoir.
    double mu_res = args.at("esn.mu-res").as<double>();
    // The upper bound for the input weights.
    // Those are generated uniformly from [0, in_upper].
    std::vector<double> in_upper = args.at("esn.in-weight").as<std::vector<double>>();
    // The upper bound for the feedback weights.
    // Those are generated uniformly from [0, fb_upper].
    std::vector<double> fb_upper = args.at("esn.fb-weight").as<std::vector<double>>();
    // The sparsity of the reservoir weight matrix. For 0, the matrix is
    // fully connected. For 1, the matrix is completely zero.
    double sparsity = args.at("esn.sparsity").as<double>();
    // Standard deviation of the noise added to the states.
    double noise = args.at("esn.noise").as<double>();
    // The leakage rate.
    double leakage = args.at("esn.leakage").as<double>();

    af::randomEngine af_prng{AF_RANDOM_ENGINE_DEFAULT, prng()};
    af::array reservoir_w = sigma_res * af::randn({n, n}, DType, af_prng) + mu_res;
    af::array input_w = af::randu({n, n_ins}, DType, af_prng);
    for (long i = 0; i < n_ins; ++i) input_w(af::span, i) *= in_upper.at(i);
    af::array feedback_w = af::randu({n, n_outs}, DType, af_prng);
    for (long i = 0; i < n_outs; ++i) input_w(af::span, i) *= fb_upper.at(i);
    // make the reservoir sparse by the given coefficient
    reservoir_w *= af::randu({reservoir_w.dims()}, DType, af_prng) >= sparsity;
    return simple_esn<DType>{n,
                             input_names,
                             output_names,
                             std::move(reservoir_w),
                             std::move(input_w),
                             std::move(feedback_w),
                             noise,
                             leakage,
                             prng};
}

/// Echo state network options description for command line parsing.
po::options_description esn_arg_description()
{
    po::options_description esn_arg_desc{"Echo state network options"};
    esn_arg_desc.add_options()                                                    //
      ("esn.neurons", po::value<long>()->default_value(128),                      //
       "The number of neurons.")                                                  //
      ("esn.sigma-res", po::value<double>()->default_value(0.19762725044833218),  //
       "See random_esn().")                                                       //
      ("esn.mu-res", po::value<double>()->default_value(-0.0068959284626413861),  //
       "See random_esn().")                                                       //
      ("esn.in-weight",                                                           //
       po::value<std::vector<double>>()                                           //
         ->multitoken()                                                           //
         ->default_value({-0.004004819844231784}, "-0.004004819844231784"),       //
       "See random_esn().")                                                       //
      ("esn.fb-weight",                                                           //
       po::value<std::vector<double>>()                                           //
         ->multitoken()                                                           //
         ->default_value({0}, "0"),                                               //
       "See random_esn().")                                                       //
      ("esn.sparsity", po::value<double>()->default_value(0),                     //
       "See random_esn().")                                                       //
      ("esn.noise", po::value<double>()->default_value(0),                        //
       "Standard deviation of the noise added to the states of the network.")     //
      ("esn.leakage", po::value<double>()->default_value(1),                      //
       "See random_esn().");                                                      //
    return esn_arg_desc;
}

}  // end namespace esn
