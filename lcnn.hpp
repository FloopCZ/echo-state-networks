#pragma once

// Locally connected Echo state network class and training functions. //

#include "arrayfire_utils.hpp"
#include "common.hpp"
#include "data_map.hpp"
#include "lcnn_adapt.hpp"
#include "lcnn_step.hpp"
#include "net.hpp"
#include "third_party/elasticnet_af/elasticnet_af.hpp"

#include <arrayfire.h>
#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <fmt/format.h>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <range/v3/all.hpp>
#include <stdexcept>

namespace esn {

namespace po = boost::program_options;

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
    // The mapping and weight of each state position to its memory point (0 to memory_length - 1).
    af::array memory_map;
    af::array memory_w;

    /// The standard deviation of the noise added to the potentials.
    double noise = 0;
    /// The leakage of the potential.
    double leakage = 1.0;
    /// The l2 regularization coefficient (only used when enet_lambda==0).
    double l2 = 0;
    /// The elastic net regularization parameters.
    double enet_lambda = 0;
    double enet_alpha = 0;
    bool enet_standardize = false;
    /// The number of intermediate steps of the network with each input.
    long intermediate_steps = 1;
    /// The number of training trials (select random indices, train, repeat).
    long n_train_trials = 1;
    /// Indices of neurons used as predictors during training.
    /// Set to 1 to use all neurons.
    double n_state_predictors = 1.;
    // How should the result of multiple calls to train() be aggregated.
    std::string train_aggregation = "ensemble";
    // The probability that a single data point belongs to the valid set during train trial.
    double train_valid_ratio = 0.8;
    // The steepness of the activation function.
    double act_steepness = 1.0;
    // Configuration for the lcnn weight adaptation.
    lcnn_adaptation_config adaptation_cfg;

    lcnn_config() = default;
    lcnn_config(const po::variables_map& args)
    {
        noise = args.at("lcnn.noise").as<double>();
        leakage = args.at("lcnn.leakage").as<double>();
        l2 = args.at("lcnn.l2").as<double>();
        enet_lambda = args.at("lcnn.enet-lambda").as<double>();
        enet_alpha = args.at("lcnn.enet-alpha").as<double>();
        enet_standardize = args.at("lcnn.enet-standardize").as<bool>();
        intermediate_steps = args.at("lcnn.intermediate-steps").as<long>();
        n_train_trials = args.at("lcnn.n-train-trials").as<long>();
        n_state_predictors = std::clamp(args.at("lcnn.n-state-predictors").as<double>(), 0., 1.);
        train_aggregation = args.at("lcnn.train-aggregation").as<std::string>();
        train_valid_ratio = args.at("lcnn.train-valid-ratio").as<double>();
        act_steepness = args.at("lcnn.act-steepness").as<double>();
        adaptation_cfg.learning_rate = args.at("lcnn.adapt.learning-rate").as<double>();
        adaptation_cfg.weight_leakage = args.at("lcnn.adapt.weight-leakage").as<double>();
        adaptation_cfg.abs_target_activation =
          args.at("lcnn.adapt.abs-target-activation").as<double>();
    }
};

/// Advanced Echo State Networks with various reservoir topologies.
template <af::dtype DType = DEFAULT_AF_DTYPE>
class lcnn : public net_base {
protected:
    std::set<std::string> input_names_;
    std::set<std::string> output_names_;
    af::array state_delta_;  // working variable used during the step function
    af::array state_;
    af::array memory_map_;
    af::array memory_w_;
    long memory_length_;
    af::array state_memory_;
    data_map last_output_;  // the last output of the net as a data map
    data_map prev_step_feedback_;
    af::array reservoir_w_;
    af::array reservoir_w_full_;
    af::array reservoir_b_;
    af::array input_w_;
    af::array feedback_w_;
    std::vector<af::array> output_w_;
    bool force_matmul_;

    // Random engines.
    prng_t prng_init_;
    prng_t prng_;

    bool noise_enabled_;
    double noise_;
    double leakage_;
    double l2_;
    double enet_lambda_;
    double enet_alpha_;
    bool enet_standardize_;
    long intermediate_steps_;
    long n_train_trials_;
    double n_state_predictors_;
    std::string train_aggregation_;
    double train_valid_ratio_;
    double act_steepness_;
    bool learning_enabled_;
    lcnn_adaptation_config adaptation_cfg;

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
        state_delta_ += update_via_weights_matmul_impl(state_);
    }

    af::array update_via_weights_impl(const af::array& state)
    {
        return lcnn_step(state, reservoir_w_);
    }

    /// Update the state matrix using the lcnn step function.
    virtual void update_via_weights()
    {
        assert(!force_matmul_);
        state_delta_ += update_via_weights_impl(state_);
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

    /// Update the state matrix by applying the activation function.
    virtual void update_via_activation()
    {
        // Add noise to the states.
        if (noise_enabled_ && noise_ != 0.) {
            af::randomEngine af_prng{AF_RANDOM_ENGINE_DEFAULT, prng_()};
            state_delta_ *= 1. + af::randn({state_.dims()}, DType, af_prng) * noise_;
        }
        // Leak some potential.
        state_ *= 1. - leakage_;
        // Apply the activation function.
        state_ += leakage_ * af::tanh(act_steepness_ * std::move(state_delta_) + reservoir_b_);
    }

    /// Update the last output of the network after having a new state.
    virtual void update_last_output()
    {
        if (output_w_.empty()) {
            last_output_.clear();
            return;
        }
        assert(
          train_aggregation_ == "replace" || train_aggregation_ == "ensemble"
          || train_aggregation_ == "delta" || train_aggregation_ == "funagg");
        af::array output =
          af::constant(af::NaN, output_names_.size(), output_w_.size(), state_.type());
        // Evaluate every output_w and aggregate them to the final output.
        for (int i = 0; i < (long)output_w_.size(); ++i) {
            const af::array& ow = output_w_.at(i);
            af::array predictors = af_utils::add_ones(af::flat(state_), 0);
            output(af::span, i) = af::matmul(ow, predictors);
            if (train_aggregation_ == "funagg") {
                if (i == 1)
                    output(af::span, i) =
                      af::atanh(af::clamp(output(af::span, i), -1, 1) * 0.99) * 30.;
                else if (i == 2)
                    output(af::span, i) =
                      af::atanh(af::clamp(output(af::span, i), -1, 1) * 0.99) * 30. - 3.;
                else if (i == 3)
                    output(af::span, i) =
                      af::atanh(af::clamp(output(af::span, i), -1, 1) * 0.99) * 30. + 3.;
            }
        }
        if (train_aggregation_ == "replace") assert(output_w_.size() == 1);
        if (train_aggregation_ == "ensemble") output = af::median(output, 1);
        if (train_aggregation_ == "delta") output = af::sum(output, 1);
        if (train_aggregation_ == "funagg") output = af::mean(output, 1);
        last_output_ = {output_names_, output};
        assert(last_output_.data().dims() == (af::dim4{(long)output_names_.size()}));
        assert(af::allTrue<bool>(!af::isNaN(af::flat(last_output_.data()))));
    }

    virtual void update_state_memory()
    {
        state_memory_ = af::shift(state_memory_, 0, 0, 1);
        state_memory_(af::span, af::span, 0) = state_;
    }

    virtual void update_via_memory()
    {
        if (memory_length_ <= 0) return;
        af::array memory = af::moddims(
          state_memory_.slices(0, memory_length_ - 1), state_.elements(), memory_length_);
        af::array state_indices = af::array(af::seq(state_.elements())).as(DType);
        memory = af::approx2(memory, state_indices, af::flat(memory_map_));
        state_delta_ += memory_w_ * af::moddims(memory, state_.dims());
    }

    void adapt_weights()
    {
        assert(!state_memory_.isempty() && state_memory_.dims(2) >= 3);
        if (!learning_enabled_ || adaptation_cfg.learning_rate == 0.) return;
        if (force_matmul_)
            throw std::runtime_error{
              "Weight adaptation implemented only for local (lcnn) reservoirs."};
        assert(!reservoir_w_.isempty());
        reservoir_w_ = lcnn_adapt(state_memory_, reservoir_w_, adaptation_cfg);
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
        af::randomEngine af_prng{AF_RANDOM_ENGINE_DEFAULT, prng_()};
        af::array idxs = af_utils::shuffle(af::seq(state_.elements()), af_prng)(af::seq(n));
        return af::sort(std::move(idxs));
    }

public:
    lcnn() = default;

    /// Locally connected echo state network constructor.
    lcnn(lcnn_config cfg, prng_t prng)
      : input_names_{cfg.input_names}
      , output_names_{cfg.output_names}
      , memory_length_{0}
      , last_output_{}
      , force_matmul_{false}
      , prng_init_{std::move(prng)}
      , prng_{prng_init_}
      , noise_enabled_{true}
      , noise_{cfg.noise}
      , leakage_{cfg.leakage}
      , l2_{cfg.l2}
      , enet_lambda_{cfg.enet_lambda}
      , enet_alpha_{cfg.enet_alpha}
      , enet_standardize_{cfg.enet_standardize}
      , intermediate_steps_{cfg.intermediate_steps}
      , n_train_trials_{cfg.n_train_trials}
      , n_state_predictors_{cfg.n_state_predictors}
      , train_aggregation_{cfg.train_aggregation}
      , train_valid_ratio_{cfg.train_valid_ratio}
      , act_steepness_(cfg.act_steepness)
      , learning_enabled_{true}
      , adaptation_cfg(cfg.adaptation_cfg)
    {
        state(std::move(cfg.init_state));
        memory_map(std::move(cfg.memory_map));
        memory_w(std::move(cfg.memory_w));
        assert(cfg.reservoir_w.isempty() ^ cfg.reservoir_w_full.isempty());
        if (!cfg.reservoir_w.isempty()) reservoir_weights(std::move(cfg.reservoir_w));
        if (!cfg.reservoir_w_full.isempty())
            reservoir_weights_full(std::move(cfg.reservoir_w_full));
        reservoir_biases(std::move(cfg.reservoir_b));
        input_weights(std::move(cfg.input_w));
        feedback_weights(std::move(cfg.feedback_w));
    }

    void save(const fs::path& dir) override
    {
        fs::create_directories(dir);

        nlohmann::json data;
        data["snapshot_version"] = 1;

        data["input_names"] = input_names_;
        data["output_names"] = output_names_;
        data["memory_length"] = memory_length_;
        data["output_w_size"] = output_w_.size();
        data["force_matmul"] = force_matmul_;
        data["noise_enabled"] = noise_enabled_;
        data["noise"] = noise_;
        data["leakage"] = leakage_;
        data["l2"] = l2_;
        data["enet_lambda"] = enet_lambda_;
        data["enet_alpha"] = enet_alpha_;
        data["enet_standardize"] = enet_standardize_;
        data["intermediate_steps"] = intermediate_steps_;
        data["n_train_trials"] = n_train_trials_;
        data["n_state_predictors"] = n_state_predictors_;
        data["train_aggregation"] = train_aggregation_;
        data["train_valid_ratio"] = train_valid_ratio_;
        data["act_steepness"] = act_steepness_;
        data["learning_enabled"] = learning_enabled_;
        data["adaptation_cfg"] = {
          {"learning_rate", adaptation_cfg.learning_rate},
          {"abs_target_activation", adaptation_cfg.abs_target_activation},
          {"weight_leakage", adaptation_cfg.weight_leakage},
        };
        std::ofstream{dir / "params.json"} << data.dump(2);

        {
            std::string p = dir / "state_delta.bin";
            af::saveArray("data", state_delta_, p.c_str());
        }
        {
            std::string p = dir / "state.bin";
            af::saveArray("data", state_, p.c_str());
        }
        if (!memory_map_.isempty()) {
            std::string p = dir / "memory_map.bin";
            af::saveArray("data", memory_map_, p.c_str());
        }
        if (!memory_w_.isempty()) {
            std::string p = dir / "memory_w.bin";
            af::saveArray("data", memory_w_, p.c_str());
        }
        if (!state_memory_.isempty()) {
            std::string p = dir / "state_memory.bin";
            af::saveArray("data", state_memory_, p.c_str());
        }
        last_output_.save(dir / "last_output/");
        prev_step_feedback_.save(dir / "prev_step_feedback/");
        {
            std::string p = dir / "reservoir_w.bin";
            af::saveArray("data", reservoir_w_, p.c_str());
        }
        if (!reservoir_w_full_.isempty()) {
            std::string p = dir / "reservoir_w_full.bin";
            af::saveArray("data", reservoir_w_full_, p.c_str());
        }
        {
            std::string p = dir / "reservoir_b.bin";
            af::saveArray("data", reservoir_b_, p.c_str());
        }
        {
            std::string p = dir / "input_w.bin";
            af::saveArray("data", input_w_, p.c_str());
        }
        {
            std::string p = dir / "feedback_w.bin";
            af::saveArray("data", feedback_w_, p.c_str());
        }
        fs::create_directories(dir / "output_w/");
        for (std::size_t i = 0; i < output_w_.size(); ++i) {
            std::string p = dir / "output_w" / (std::to_string(i) + ".bin");
            af::saveArray("data", output_w_.at(i), p.c_str());
        }

        std::ofstream{dir / "prng_init.bin"} << prng_init_;
        std::ofstream{dir / "prng.bin"} << prng_;
    }

    static lcnn<DType> load(const fs::path& dir)
    {
        if (!fs::exists(dir))
            throw std::runtime_error{"LCNN snapshot dir `" + dir.string() + "` does not exist."};
        if (!fs::exists(dir / "params.json"))
            throw std::runtime_error{
              "Dir `" + dir.string() + "` does not seem to be a LCNN snapshot directory."};

        lcnn<DType> net;
        nlohmann::json data = nlohmann::json::parse(std::ifstream{dir / "params.json"});
        if (data.at("snapshot_version").get<int>() != 1)
            throw std::runtime_error{"Snapshot not compatible with the current binary."};

        net.input_names_ = data["input_names"];
        net.output_names_ = data["output_names"];
        net.memory_length_ = data["memory_length"];
        net.force_matmul_ = data["force_matmul"];
        net.noise_enabled_ = data["noise_enabled"];
        net.noise_ = data["noise"];
        net.leakage_ = data["leakage"];
        net.l2_ = data["l2"];
        // TODO remove defaults when all the models are updated
        net.enet_lambda_ = data.value("enet_lambda", 0.);
        net.enet_alpha_ = data.value("enet_alpha", 0.);
        net.enet_standardize_ = data.value("enet_standardize", false);
        net.intermediate_steps_ = data["intermediate_steps"];
        net.n_train_trials_ = data["n_train_trials"];
        net.n_state_predictors_ = data["n_state_predictors"];
        net.train_aggregation_ = data["train_aggregation"];
        net.train_valid_ratio_ = data["train_valid_ratio"];
        net.act_steepness_ = data["act_steepness"];
        net.learning_enabled_ = data["learning_enabled"];
        net.adaptation_cfg.learning_rate = data["adaptation_cfg"]["learning_rate"];
        net.adaptation_cfg.abs_target_activation = data["adaptation_cfg"]["abs_target_activation"];
        net.adaptation_cfg.weight_leakage = data["adaptation_cfg"]["weight_leakage"];

        {
            std::string p = dir / "state_delta.bin";
            net.state_delta_ = af::readArray(p.c_str(), "data");
        }
        {
            std::string p = dir / "state.bin";
            net.state_ = af::readArray(p.c_str(), "data");
        }
        {
            std::string p = dir / "memory_map.bin";
            if (fs::exists(p)) net.memory_map_ = af::readArray(p.c_str(), "data");
        }
        {
            std::string p = dir / "memory_w.bin";
            if (fs::exists(p)) net.memory_w_ = af::readArray(p.c_str(), "data");
        }
        {
            std::string p = dir / "state_memory.bin";
            if (fs::exists(p)) net.state_memory_ = af::readArray(p.c_str(), "data");
        }
        net.last_output_.load(dir / "last_output/");
        net.prev_step_feedback_.load(dir / "prev_step_feedback/");
        {
            std::string p = dir / "reservoir_w.bin";
            net.reservoir_w_ = af::readArray(p.c_str(), "data");
        }
        {
            std::string p = dir / "reservoir_w_full.bin";
            if (fs::exists(p)) net.reservoir_w_full_ = af::readArray(p.c_str(), "data");
        }
        {
            std::string p = dir / "reservoir_b.bin";
            net.reservoir_b_ = af::readArray(p.c_str(), "data");
        }
        {
            std::string p = dir / "input_w.bin";
            net.input_w_ = af::readArray(p.c_str(), "data");
        }
        {
            std::string p = dir / "feedback_w.bin";
            net.feedback_w_ = af::readArray(p.c_str(), "data");
        }

        std::size_t output_w_size = data["output_w_size"];
        net.output_w_.clear();
        for (std::size_t i = 0; i < output_w_size; ++i) {
            std::string p = dir / "output_w" / (std::to_string(i) + ".bin");
            net.output_w_.push_back(af::readArray(p.c_str(), "data"));
        }

        std::ifstream{dir / "prng_init.bin"} >> net.prng_init_;
        std::ifstream{dir / "prng.bin"} >> net.prng_;
        return net;
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
      const data_map& step_meta,
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
        if (!step_feedback.empty()) {
            assert(step_feedback.length() == 1);
        }
        if (!step_desired.empty()) {
            assert(step_desired.length() == 1);
            assert(step_desired.keys() == output_names_);
        }

        // Prepare state delta.
        state_delta_ = af::constant(0, state_.dims(), state_.type());

        // Update the internal state.
        for (long interm_step = 0; interm_step < intermediate_steps_; ++interm_step) {
            // Perform matrix multiplication instead of state unwrapping for large kernels.
            if (do_matmul_step()) {
                update_via_weights_matmul();
            } else {
                // Use state unwrapping for small kernels
                update_via_weights();
            }

            // Restore memory neuron states from memory.
            update_via_memory();

            // add input
            update_via_input(tr_step_input.data());

            // add feedback
            if (!tr_last_output.empty()) {
                assert(tr_last_output.keys() == output_names_);
                update_via_feedback(tr_last_output.data());
            }

            // activation function
            update_via_activation();

            update_state_memory();

            adapt_weights();

            assert(!af::anyTrue<bool>(af::isNaN(state_)));
        }

        update_last_output();

        // Call the registered callback functions.
        for (on_state_change_callback_t& fnc : on_state_change_callbacks_) {
            on_state_change_data data = {
              .state = state_,
              .input =
                {.input = step_input,
                 .feedback = step_feedback,
                 .desired = step_desired,
                 .meta = step_meta,
                 .input_transform = input_transform},
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
        assert(data_len > 0);

        feed_result_t result;
        result.states = af::array(state_.dims(0), state_.dims(1), data_len, DType);
        result.outputs = af::constant(af::NaN, output_names_.size(), data_len, DType);
        result.desired = input.desired.data();
        for (long i = 0; i < data_len; ++i) {
            // prepare the inputs for this step
            data_map step_input = input.input.select(i);
            data_map step_feedback = input.feedback.select(i);
            data_map step_desired = input.desired.select(i);
            data_map step_meta = input.meta.select(i);
            step(step_input, step_feedback, step_desired, step_meta, input.input_transform);
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
        return train(feed(input), input);
    }

    /// Clear the network output weights and reset prng to the initial state.
    void reset() override
    {
        prng_ = prng_init_;
        output_w_.clear();
    }

    /// Train the network on already processed feed result.
    train_result_t train_impl(
      const feed_result_t& data,
      const af::array& state_predictor_indices,
      const af::array& training_weights) const
    {
        assert(data.states.type() == DType);
        assert(
          (data.states.dims() == af::dim4{state_.dims(0), state_.dims(1), data.outputs.dims(1)}));
        assert(data.outputs.type() == DType);
        assert(data.outputs.numdims() <= 2);
        assert(data.outputs.dims(0) == (long)output_names_.size());
        if (!data.desired) throw std::runtime_error{"No desired data to train to."};
        assert(data.outputs.dims(1) == data.desired->dims(1));
        assert(data.desired->type() == DType);
        assert(data.desired->numdims() <= 2);
        assert(data.desired->dims(0) == (long)output_names_.size());
        assert(!af::anyTrue<bool>(af::isNaN(data.states)));
        assert(!af::anyTrue<bool>(af::isNaN(*data.desired)));
        af::array flat_predictors =
          af::moddims(data.states, state_.elements(), data.desired->dims(1)).T();
        af::array predictors = flat_predictors;
        if (!state_predictor_indices.isempty())
            predictors = flat_predictors(af::span, state_predictor_indices);
        // Find the regression coefficients.
        af::array beta = af::constant(0., output_names_.size(), state_.elements() + 1, DType);
        if (enet_lambda_ == 0.) {
            beta = af_utils::lstsq_train(predictors, data.desired->T(), l2_).T();
        } else {
            elasticnet_af::ElasticNet enet{
              {.lambda = enet_lambda_,
               .alpha = enet_alpha_,
               .tol = 1e-12,
               .path_len = 1,
               .max_grad_steps = 100,
               .standardize_var = enet_standardize_,
               .warm_start = true}};
            try {
                enet.fit(
                  predictors, data.desired->T(), training_weights);  // Ignore failed convergence.
                beta = enet.coefficients(true).T();
            } catch (const std::invalid_argument& e) {
                std::cerr << "Invalid input to ElasticNet: " << e.what() << std::endl;
            }
        }
        // Distribute the coefficients along the state_predictor_indices, leave the other empty.
        af::array output_w = [&]() {
            if (state_predictor_indices.isempty()) return beta;
            af::array w =
              af::constant(0., output_names_.size(), state_.elements() + 1, flat_predictors.type());
            af::array output_w_indices = af_utils::add_zeros(state_predictor_indices + 1, 0);
            w(af::span, output_w_indices) = beta;
            w(af::isNaN(w) || af::isInf(w)) = 0.;
            return w;
        }();
        assert(
          output_w.dims() == (af::dim4{(long)output_names_.size(), (long)state_.elements() + 1}));
        return {.predictors = std::move(flat_predictors), .output_w = std::move(output_w)};
    }

    /// Train the network on already processed feed result.
    /// \param data Training data.
    train_result_t train(feed_result_t data, const input_t& input) override
    {
        if (!data.desired) throw std::runtime_error{"No desired data to train to."};

        if (!output_w_.empty()) {
            double feed_err = af_utils::mse<double>(data.outputs, *data.desired);
            std::cout << fmt::format(
              "Before train {} MSE error (all): {}", output_w_.size(), feed_err)
                      << std::endl;
        }

        assert(
          train_aggregation_ == "replace" || train_aggregation_ == "ensemble"
          || train_aggregation_ == "delta" || train_aggregation_ == "funagg");
        // Prepare for delta training (epoch 1 and later).
        // In the second and later epochs, we only train the difference.
        if (train_aggregation_ == "delta" && !output_w_.empty())
            data.desired = *data.desired - data.outputs;
        if (train_aggregation_ == "funagg") {
            if (output_w_.size() == 1)
                data.desired = af::tanh(*data.desired / 30.) / 0.99;
            else if (output_w_.size() == 2)
                data.desired = af::tanh((*data.desired + 3.) / 30.) / 0.99;
            else if (output_w_.size() == 3)
                data.desired = af::tanh((*data.desired - 3.) / 30.) / 0.99;
        }

        struct train_trial_result_t {
            train_result_t result;
            double valid_err;
        } best_train{.result = {}, .valid_err = std::numeric_limits<double>::max()};

        long n_predictors = std::clamp(
          std::lround(n_state_predictors_ * state_.elements()), 1L, (long)state_.elements());
        bool predictor_subset = n_predictors < state_.elements();
        bool cross_validate = predictor_subset && n_train_trials_ > 1;

        // split the data to train/valid if not using all state neurons or there is just a single
        // train trial
        feed_result_t train_data = data;
        feed_result_t valid_data = data;
        if (cross_validate) {
            if (train_valid_ratio_ <= 0. || train_valid_ratio_ >= 1.)
                throw std::invalid_argument{
                  fmt::format("Invalid lcnn.train-valid-ratio {}", train_valid_ratio_)};
            af::array train_set_idx;
            af::array valid_set_idx;
            long train_count = 0;
            af::randomEngine af_prng{AF_RANDOM_ENGINE_DEFAULT, prng_()};
            while (train_count < 2 || train_count > train_set_idx.elements() - 2) {
                train_set_idx =
                  af::randu(data.states.dims(2), af::dtype::f32, af_prng) < train_valid_ratio_;
                valid_set_idx = !train_set_idx;
                train_count = af::count<long>(train_set_idx);
            }
            train_data = {
              .states = data.states(af::span, af::span, train_set_idx),
              .outputs = data.outputs(af::span, train_set_idx),
              .desired = (*data.desired)(af::span, train_set_idx)};
            valid_data = {
              .states = data.states(af::span, af::span, valid_set_idx),
              .outputs = data.outputs(af::span, valid_set_idx),
              .desired = (*data.desired)(af::span, valid_set_idx)};
            data = {};  // free memory
        }

        assert(n_train_trials_ > 0);
        for (long i = 0; i < n_train_trials_; ++i) {
            // select random state predictor indices
            af::array state_predictor_indices;
            if (predictor_subset)
                state_predictor_indices = generate_random_state_indices(n_predictors);
            // set exponential training weights
            long n = train_data.states.dims(2);
            af::array seq = af::seq(n);
            af::array training_weights = af::exp(seq.as(DType) / n);
            // train
            train_result_t train_result =
              train_impl(train_data, state_predictor_indices, training_weights);
            af::array train_prediction =
              af_utils::lstsq_predict(train_result.predictors, train_result.output_w.T());
            double train_err = af_utils::mse<double>(train_prediction.T(), *train_data.desired);
            double valid_err = std::numeric_limits<double>::quiet_NaN();
            // predict out of sample
            if (cross_validate) {
                af::array valid_predictors =
                  af::moddims(valid_data.states, state_.elements(), valid_data.states.dims(2));
                af::array valid_prediction =
                  af_utils::lstsq_predict(valid_predictors.T(), train_result.output_w.T());
                valid_err = af_utils::mse<double>(valid_prediction.T(), *valid_data.desired);
            }
            // print statistics
            std::cout << fmt::format(
              "Train {} trial {} MSE error (train): {}", output_w_.size(), i, train_err)
                      << std::endl;
            if (cross_validate)
                std::cout << fmt::format(
                  "Train {} trial {} MSE error (valid): {}", output_w_.size(), i, valid_err)
                          << std::endl;
            // select the best train trial
            if (std::isnan(valid_err) || valid_err < best_train.valid_err)
                best_train = {.result = std::move(train_result), .valid_err = valid_err};
            // no need to keep on trying if use all state predictors (the result will be the same)
            if (!cross_validate) break;
        }

        if (train_aggregation_ == "replace") output_w_.clear();
        output_w_.push_back(best_train.result.output_w);
        update_last_output();
        return best_train.result;
    }

    /// Clear the stored feedback which would otherwise be used in the next step.
    void clear_feedback() override
    {
        prev_step_feedback_.clear();
    }

    /// Get the current output weights.
    const std::vector<af::array>& output_w() const
    {
        return output_w_;
    }

    /// Set the output weights.
    void output_w(std::vector<af::array> output_w)
    {
        output_w_ = std::move(output_w);
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
          (new_weights.dims()
           == af::dim4{state_.dims(0), state_.dims(1), (long)input_names_.size()}));
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
          (new_weights.dims()
           == af::dim4{state_.dims(0), state_.dims(1), (long)output_names_.size()}));
        feedback_w_ = std::move(new_weights);
    }

    /// Get the feedback weights of the network.
    const af::array& feedback_weights() const
    {
        return feedback_w_;
    }

    /// Set the memory map (and memory length) of the network.
    ///
    /// The shape has to be the same as the state.
    void memory_map(af::array new_map)
    {
        if (new_map.isempty()) {
            memory_length_ = 0;
        } else {
            assert(new_map.type() == DType);
            assert(new_map.dims() == state_.dims());
            memory_map_ = std::move(new_map);
            memory_length_ = std::roundl(af::max<double>(memory_map_)) + 1;
        }
        // We need memory of length at least 3 for weight adaptation.
        state_memory_ = af::tile(state_, 1, 1, std::max(3L, memory_length_));
    }

    /// Set the memory weights of the network.
    ///
    /// The shape has to be the same as the state.
    void memory_w(af::array new_weights)
    {
        assert(new_weights.type() == DType);
        assert(new_weights.dims() == state_.dims());
        memory_w_ = std::move(new_weights);
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
            af::randomEngine af_prng{AF_RANDOM_ENGINE_DEFAULT, prng_()};
            af::array rand_state = af::randu({state_.dims()}, DType, af_prng) / state_.elements();
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

    /// Disable random noise e.g., for lyapunov testing.
    void random_noise(bool enable) override
    {
        noise_enabled_ = enable;
    }

    /// Disable weight adaptation.
    void learning(bool enable) override
    {
        learning_enabled_ = enable;
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
  prng_t& prng)
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
    // The input weight and bias for each input.
    std::vector<double> mu_in_weight = args.at("lcnn.mu-in-weight").as<std::vector<double>>();
    if ((long)mu_in_weight.size() < n_ins)
        throw std::invalid_argument{
          fmt::format("mu-in-weight ({}) < n_ins ({})", mu_in_weight.size(), n_ins)};
    std::vector<double> sigma_in_weight = args.at("lcnn.sigma-in-weight").as<std::vector<double>>();
    if ((long)sigma_in_weight.size() < n_ins)
        throw std::invalid_argument{
          fmt::format("sigma-in-weight ({}) < n_ins ({})", sigma_in_weight.size(), n_ins)};
    // The feedback weight and bias for each output.
    std::vector<double> mu_fb_weight = args.at("lcnn.mu-fb-weight").as<std::vector<double>>();
    if ((long)mu_fb_weight.size() < n_outs)
        throw std::invalid_argument{
          fmt::format("mu-fb-weight ({}) < n_outs ({})", mu_fb_weight.size(), n_outs)};
    std::vector<double> sigma_fb_weight = args.at("lcnn.sigma-fb-weight").as<std::vector<double>>();
    if ((long)sigma_fb_weight.size() < n_outs)
        throw std::invalid_argument{
          fmt::format("sigma-fb-weight ({}) < n_outs ({})", sigma_fb_weight.size(), n_outs)};
    // Standard deviation of the normal distribution generating the biases.
    double sigma_b = args.at("lcnn.sigma-b").as<double>();
    // The mean of the normal distribution generating the biases.
    double mu_b = args.at("lcnn.mu-b").as<double>();
    // The sparsity of the reservoir weight matrix. For 0, the matrix is
    // fully connected. For 1, the matrix is completely zero.
    double sparsity = args.at("lcnn.sparsity").as<double>();
    // The sparsity of the input and feedback matrix.
    double in_fb_sparsity = args.at("lcnn.in-fb-sparsity").as<double>();
    // The reservoir topology.
    std::string topology = args.at("lcnn.topology").as<std::string>();
    std::set<std::string> topo_params;
    boost::split(topo_params, topology, boost::is_any_of("-"));
    // How many neurons are injected with each input.
    double input_to_n = std::clamp(args.at("lcnn.input-to-n").as<double>(), 0., 1.);
    // The maximum memory length.
    long memory_length = std::clamp(args.at("lcnn.memory-length").as<long>(), 0L, 1000L);
    // The probability that a neuron is a memory neuron.
    double memory_prob = std::clamp(args.at("lcnn.memory-prob").as<double>(), 0., 1.);
    // The distribution of memory weights.
    double sigma_memory = args.at("lcnn.sigma-memory").as<double>();
    double mu_memory = args.at("lcnn.mu-memory").as<double>();

    if (kernel_height % 2 == 0 || kernel_width % 2 == 0)
        throw std::invalid_argument{"Kernel size has to be odd."};

    lcnn_config cfg{args};
    cfg.input_names = input_names;
    cfg.output_names = output_names;
    af::randomEngine af_prng{AF_RANDOM_ENGINE_DEFAULT, prng()};
    int neurons = state_height * state_width;
    int half_kernel_height = kernel_height / 2;
    int half_kernel_width = kernel_width / 2;
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
            kernel(af::span, af::seq(half_kernel_width, af::end)) = 0.;
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
        af::dim4 res_dims{state_height, state_width, kernel_height, kernel_width};
        cfg.reservoir_w = sigma_res * (af::randu(res_dims, DType, af_prng) * 2 - 1) + mu_res;
        // make the reservoir sparse by the given coefficient
        cfg.reservoir_w *= af::randu({cfg.reservoir_w.dims()}, DType, af_prng) >= sparsity;
        if (topo_params.contains("noself")) {
            cfg.reservoir_w(af::span, af::span, half_kernel_height, half_kernel_width) = 0.;
        }
        if (topo_params.contains("od")) {
            // only allow connections going to the right
            cfg.reservoir_w(af::span, af::span, af::span, af::seq(half_kernel_width, af::end)) = 0.;
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
            cfg.reservoir_w(af::span, af::span, af::span, af::seq(half_kernel_width, af::end)) = 0.;
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
            cfg.reservoir_w(af::span, i, af::span, af::seq(0, half_kernel_width - i - 1)) = 0;
            cfg.reservoir_w(
              af::span, state_width - i - 1, af::span,
              af::seq(half_kernel_width + i + 1, af::end)) = 0;
        }
        for (int i = 0; i < kernel_height / 2; ++i) {
            cfg.reservoir_w(i, af::span, af::seq(0, half_kernel_height - i - 1), af::span) = 0;
            cfg.reservoir_w(
              state_height - i - 1, af::span, af::seq(half_kernel_height + i + 1, af::end),
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

    long n_input_neurons = std::clamp(
      std::lround(input_to_n * state_height * state_width), 1L, state_height * state_width);
    if (n_input_neurons == state_height * state_width) {
        // put input and feedback into all the neurons
        cfg.input_w = af::randu({state_height, state_width, n_ins}, DType, af_prng) * 2 - 1;
        for (long i = 0; i < n_ins; ++i) {
            cfg.input_w(af::span, af::span, i) *= sigma_in_weight.at(i);
            cfg.input_w(af::span, af::span, i) += mu_in_weight.at(i);
        }
        cfg.input_w *= af::randu({cfg.input_w.dims()}, DType, af_prng) >= in_fb_sparsity;
        cfg.feedback_w = af::randu({state_height, state_width, n_outs}, DType, af_prng) * 2 - 1;
        for (long i = 0; i < n_outs; ++i) {
            cfg.feedback_w(af::span, af::span, i) *= sigma_fb_weight.at(i);
            cfg.feedback_w(af::span, af::span, i) += mu_fb_weight.at(i);
        }
        cfg.feedback_w *= af::randu({cfg.feedback_w.dims()}, DType, af_prng) >= in_fb_sparsity;
    } else {
        // choose the locations for inputs and feedbacks
        cfg.input_w = af::constant(0, state_height, state_width, n_ins, DType);
        for (long i = 0; i < n_ins; ++i) {
            if (n_input_neurons == 1 && free_position != nice_positions.end()) {
                cfg.input_w(free_position->first, free_position->second, i) = mu_in_weight.at(i);
                ++free_position;
            } else {
                af::array input_w_single = af::constant(0, state_height, state_width, DType);
                af::array idxs = af_utils::shuffle(af::seq(state_height * state_width), af_prng)(
                  af::seq(n_input_neurons));
                input_w_single(idxs) =
                  (af::randu(n_input_neurons, DType, af_prng) * 2 - 1) * sigma_in_weight.at(i)
                  + mu_in_weight.at(i);
                cfg.input_w(af::span, af::span, i) = input_w_single;
            }
        }
        cfg.feedback_w = af::constant(0, state_height, state_width, n_outs, DType);
        for (long i = 0; i < n_outs; ++i) {
            if (n_input_neurons == 1 && free_position != nice_positions.end()) {
                cfg.feedback_w(free_position->first, free_position->second, i) = mu_fb_weight.at(i);
                ++free_position;
            } else {
                af::array feedback_w_single = af::constant(0, state_height, state_width, DType);
                af::array idxs = af_utils::shuffle(af::seq(state_height * state_width), af_prng)(
                  af::seq(n_input_neurons));
                feedback_w_single(idxs) =
                  (af::randu(n_input_neurons, DType, af_prng) * 2 - 1) * sigma_fb_weight.at(i)
                  + mu_fb_weight.at(i);
                cfg.feedback_w(af::span, af::span, i) = feedback_w_single;
            }
        }
    }

    // the initial state is full of zeros
    cfg.init_state = af::constant(0, state_height, state_width, DType);
    // the initial state is random
    // cfg.init_state = af::randu({state_height, state_width}, DType, af_prng) * 2. - 1.;

    if (memory_length == 0) {
        cfg.memory_map = af::array{};
        cfg.memory_w = af::array{};
    } else {
        cfg.memory_map = af::constant(0, {state_height, state_width}, DType);
        af::array memory_full =
          af::round(af::randu(cfg.memory_map.dims(), DType, af_prng) * (memory_length - 1));
        af::array memory_mask = af::randu(cfg.memory_map.dims()) < memory_prob;
        cfg.memory_map(memory_mask) = memory_full(memory_mask);

        cfg.memory_w = af::constant(0, {state_height, state_width}, DType);
        af::array memory_w_full =
          (af::randu(cfg.memory_map.dims(), DType, af_prng) * 2 - 1) * sigma_memory + mu_memory;
        cfg.memory_w(memory_mask) = memory_w_full(memory_mask);
    }

    return lcnn<DType>{std::move(cfg), prng};
}

/// Locally connected network options description for command line parsing.
inline po::options_description lcnn_arg_description()
{
    po::options_description lcnn_arg_desc{"Locally connected network options"};
    lcnn_arg_desc.add_options()                                                       //
      ("lcnn.state-height", po::value<long>()->default_value(11),                     //
       "The fixed height of the kernel.")                                             //
      ("lcnn.state-width", po::value<long>()->default_value(11),                      //
       "The width of the state matrix.")                                              //
      ("lcnn.kernel-height", po::value<long>()->default_value(5),                     //
       "The height of the kernel.")                                                   //
      ("lcnn.kernel-width", po::value<long>()->default_value(5),                      //
       "The width of the kernel.")                                                    //
      ("lcnn.sigma-res", po::value<double>()->default_value(0.2),                     //
       "See random_lcnn().")                                                          //
      ("lcnn.mu-res", po::value<double>()->default_value(0),                          //
       "See random_lcnn().")                                                          //
      ("lcnn.mu-in-weight",                                                           //
       po::value<std::vector<double>>()                                               //
         ->multitoken()                                                               //
         ->default_value(std::vector<double>{0.0}, "0.0"),                            //
       "See random_lcnn().")                                                          //
      ("lcnn.sigma-in-weight",                                                        //
       po::value<std::vector<double>>()                                               //
         ->multitoken()                                                               //
         ->default_value(std::vector<double>{0.0}, "0.0"),                            //
       "See random_lcnn().")                                                          //
      ("lcnn.mu-fb-weight",                                                           //
       po::value<std::vector<double>>()                                               //
         ->multitoken()                                                               //
         ->default_value(std::vector<double>{0}, "0"),                                //
       "See random_lcnn().")                                                          //
      ("lcnn.sigma-fb-weight",                                                        //
       po::value<std::vector<double>>()                                               //
         ->multitoken()                                                               //
         ->default_value(std::vector<double>{0}, "0.0"),                              //
       "See random_lcnn().")                                                          //
      ("lcnn.sigma-b", po::value<double>()->default_value(0),                         //
       "See random_lcnn().")                                                          //
      ("lcnn.mu-b", po::value<double>()->default_value(0),                            //
       "See random_lcnn().")                                                          //
      ("lcnn.sparsity", po::value<double>()->default_value(0),                        //
       "See random_lcnn().")                                                          //
      ("lcnn.in-fb-sparsity", po::value<double>()->default_value(0),                  //
       "See random_lcnn().")                                                          //
      ("lcnn.topology", po::value<std::string>()->default_value("sparse"),            //
       "See random_lcnn().")                                                          //
      ("lcnn.input-to-n", po::value<double>()->default_value(1.),                     //
       "See random_lcnn().")                                                          //
      ("lcnn.noise", po::value<double>()->default_value(0),                           //
       "See lcnn_config class.")                                                      //
      ("lcnn.leakage", po::value<double>()->default_value(1),                         //
       "See lcnn_config class.")                                                      //
      ("lcnn.intermediate-steps", po::value<long>()->default_value(1),                //
       "See lcnn_config class.")                                                      //
      ("lcnn.l2", po::value<double>()->default_value(0),                              //
       "See lcnn_config class.")                                                      //
      ("lcnn.enet-lambda", po::value<double>()->default_value(0),                     //
       "See lcnn_config class.")                                                      //
      ("lcnn.enet-alpha", po::value<double>()->default_value(0),                      //
       "See lcnn_config class.")                                                      //
      ("lcnn.enet-standardize", po::value<bool>()->default_value(false),              //
       "See lcnn_config class.")                                                      //
      ("lcnn.n-train-trials", po::value<long>()->default_value(1),                    //
       "See random_lcnn().")                                                          //
      ("lcnn.n-state-predictors", po::value<double>()->default_value(1.),             //
       "What fraction of neurons is used for regression training.")                   //
      ("lcnn.train-aggregation", po::value<std::string>()->default_value("replace"),  //
       "See lcnn_config class.")                                                      //
      ("lcnn.train-valid-ratio", po::value<double>()->default_value(0.8),             //
       "See lcnn_config class.")                                                      //
      ("lcnn.act-steepness", po::value<double>()->default_value(1.0),                 //
       "See lcnn_config class.")                                                      //
      ("lcnn.memory-length", po::value<long>()->default_value(0.0),                   //
       "The maximum reach of the memory. Set to 0 to disable memory.")                //
      ("lcnn.memory-prob", po::value<double>()->default_value(0.0),                   //
       "The probability that a neuron is a memory neuron.")                           //
      ("lcnn.sigma-memory", po::value<double>()->default_value(0.0),                  //
       "See random_lcnn().")                                                          //
      ("lcnn.mu-memory", po::value<double>()->default_value(1.0),                     //
       "See random_lcnn().")                                                          //
      ("lcnn.adapt.learning-rate", po::value<double>()->default_value(0.0),           //
       "Learning rate for weight adaptation. Set to 0 to disable learning.")          //
      ("lcnn.adapt.weight-leakage", po::value<double>()->default_value(0.0),          //
       "Decay rate for weight adaptation.")                                           //
      ("lcnn.adapt.abs-target-activation", po::value<double>()->default_value(1.0),   //
       "Target value of neuron activation during adaptation.")                        //
      ("lcnn.load", po::value<std::string>(),                                         //
       "Directory from which to load the network.")                                   //
      ;
    return lcnn_arg_desc;
}

}  // namespace esn
