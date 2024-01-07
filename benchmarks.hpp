#pragma once

// The benchmarks used in combination with \ref optimizer. //

#include "analysis.hpp"
#include "argument_utils.hpp"
#include "arrayfire_utils.hpp"
#include "data_map.hpp"
#include "misc.hpp"
#include "net.hpp"

#include <af/data.h>
#include <boost/algorithm/string.hpp>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <range/v3/view.hpp>
#include <vector>

namespace esn {

namespace fs = std::filesystem;
namespace po = boost::program_options;
namespace rg = ranges;
namespace rgv = ranges::views;

/// Base class for echo state network benchmarking tasks.
class benchmark_set_base {
protected:
    po::variables_map config_;
    std::vector<long> split_sizes_;
    std::string error_measure_;
    long n_epochs_;

    virtual data_map input_transform(const data_map& xs) const
    {
        return xs;
    }

    net_base::input_transform_fn_t input_transform_fn() const
    {
        return [this](const data_map& dm) -> data_map { return this->input_transform(dm); };
    }

    virtual double error_fnc(const data_map& predicted, const data_map& desired) const
    {
        assert(predicted.keys() == desired.keys());
        const af::array& predicted_arr = predicted.data();
        const af::array& desired_arr = desired.data();
        assert(predicted_arr.dims() == desired_arr.dims());
        assert(desired_arr.numdims() <= 2);
        if (error_measure_ == "mse") return af_utils::mse<double>(predicted_arr, desired_arr);
        if (error_measure_ == "nmse") return af_utils::nmse<double>(predicted_arr, desired_arr);
        if (error_measure_ == "nrmse") return af_utils::nrmse<double>(predicted_arr, desired_arr);
        throw std::invalid_argument("Unknown error measure `" + error_measure_ + "`.");
    }

    double multi_error_fnc(
      const std::vector<data_map>& predicted, const std::vector<data_map>& desired) const
    {
        assert(predicted.size() == desired.size());
        double error_sum = 0;
        for (const auto& [dmp, dmd] : rgv::zip(predicted, desired))
            error_sum += error_fnc(dmp, dmd);
        return error_sum / predicted.size();
    }

public:
    benchmark_set_base(po::variables_map config)
      : config_{std::move(config)}
      , split_sizes_{
            config_.at("bench.init-steps").as<long>()
          , config_.at("bench.train-steps").as<long>()
          , config_.at("bench.valid-steps").as<long>()}
      , error_measure_{config_.at("bench.error-measure").as<std::string>()}
      , n_epochs_{config_.at("bench.n-epochs").as<long>()}
    {
    }

    virtual double evaluate(net_base& net, std::mt19937& prng) const = 0;
    virtual const std::set<std::string>& input_names() const = 0;
    virtual const std::set<std::string>& output_names() const = 0;
    virtual ~benchmark_set_base() = default;
    virtual bool constant_data() const
    {
        return false;
    }
};

/// TODO fix docs
/// Base class for echo state network benchmarking tasks.
class markov_benchmark_set : public benchmark_set_base {
protected:
    long n_steps_ahead_;

    /// Generate the training inputs and outputs of dimensions [n_ins, len] and [n_outs, len].
    virtual std::tuple<data_map, data_map>
    generate_data(long len, af::dtype dtype, std::mt19937& prng) const = 0;

    static std::set<std::string>
    make_shifted_seq_names(const std::set<std::string>& prefixes, long n)
    {
        std::set<std::string> names;
        for (const std::string& p : prefixes)
            for (long i = 1; i <= n; ++i) names.emplace(p + "-" + std::to_string(i));
        return names;
    }

    static data_map make_shifted_seqs(const data_map& data, long n)
    {
        std::map<std::string, af::array> shifted_seqs;
        for (const std::string& name : data.keys()) {
            for (long i = 1; i <= n; ++i) {
                af::array shifted = af::shift(data.at(name), -i);
                shifted(af::seq(af::end - i, af::end)) = af::NaN;
                shifted_seqs.emplace(name + "-" + std::to_string(i), std::move(shifted));
            }
        }
        return shifted_seqs;
    }

    mutable std::mutex data_cache_mutex_;
    mutable std::unique_ptr<data_map> xs_cached_, ys_cached_, ys_shifted_cached_;

public:
    using benchmark_set_base::benchmark_set_base;

    markov_benchmark_set(po::variables_map config)
      : benchmark_set_base{std::move(config)}
      , n_steps_ahead_{config_.at("bench.n-steps-ahead").as<long>()}
    {
    }

    double evaluate(net_base& net, std::mt19937& prng) const override
    {
        long len = rg::accumulate(split_sizes_, 0L);
        // load data or initialize/load cache
        data_map xs, ys;
        data_map ys_shifted;
        if (constant_data()) {
            std::unique_lock ul{data_cache_mutex_};
            if (!xs_cached_ || !ys_cached_ || !ys_shifted_cached_) {
                std::tie(xs, ys) = generate_data(len, net.state().type(), prng);
                ys_shifted = make_shifted_seqs(ys, n_steps_ahead_);
                xs_cached_ = std::make_unique<data_map>(std::move(xs));
                ys_cached_ = std::make_unique<data_map>(std::move(ys));
                ys_shifted_cached_ = std::make_unique<data_map>(std::move(ys_shifted));
            }
            std::tie(xs, ys) = std::tie(*xs_cached_, *ys_cached_);
            ys_shifted = *ys_shifted_cached_;
        } else {
            std::tie(xs, ys) = generate_data(len, net.state().type(), prng);
            ys_shifted = make_shifted_seqs(ys, n_steps_ahead_);
        }
        // check data
        assert(xs.keys() == input_names());
        assert(xs.keys() == net.input_names());
        assert(xs.length() >= len);
        assert(ys_shifted.keys() == output_names());
        assert(ys_shifted.keys() == net.output_names());
        assert(ys_shifted.length() >= len);
        assert(ys.keys() == target_names());
        assert(ys.length() >= len);
        // split both input and output to init, train, test
        std::vector<data_map> xs_groups = xs.split(split_sizes_);
        std::vector<data_map> ys_shifted_groups = ys_shifted.split(split_sizes_);
        std::vector<data_map> ys_groups = ys.split(split_sizes_);
        double error = std::numeric_limits<double>::quiet_NaN();
        for (long epoch = 0; epoch < n_epochs_; ++epoch) {
            // initialize the network using the initial sequence
            net.event("init-start");
            net.random_noise(true);
            net.feed(
              {.input = xs_groups.at(0),
               .feedback = ys_shifted_groups.at(0),
               .desired = ys_shifted_groups.at(0),
               .input_transform = input_transform_fn()});
            // train the network on the training sequence
            net.event("train-start");
            {
                train_result_t train_result = net.train(
                  {.input = xs_groups.at(1),
                   .feedback = {},
                   .desired = ys_shifted_groups.at(1),
                   .input_transform = input_transform_fn()});
                // Print train mse error
                af::array train_prediction =
                  af::matmul(train_result.predictors, train_result.output_w.T());
                double err =
                  af_utils::mse<double>(train_prediction.T(), ys_shifted_groups.at(1).data());
                std::cout << "Train MSE error: " << err << std::endl;
            }
            net.random_noise(false);
            net.clear_feedback();
            // evaluate the performance of the network on the validation sequence
            net.event("validation-start");
            long validation_size = xs_groups.at(2).length();
            af::seq valid_selector(validation_size - n_steps_ahead_);
            feed_result_t feed_result = [&]() {
                data_map input = xs_groups.at(2).select(valid_selector);
                data_map desired = ys_shifted_groups.at(2).select(valid_selector);
                return net.feed(
                  {.input = input,
                   .feedback = {},
                   .desired = desired,
                   .input_transform = input_transform_fn()});
            }();
            data_map output{net.output_names(), feed_result.outputs};

            // build the subsequences of length n_steps_ahead_ for each time point
            std::vector<data_map> predicted;
            std::vector<data_map> desired;
            std::map<std::string, af::array> last_shift_predicted;
            std::map<std::string, af::array> last_shift_desired;
            for (const std::string& target_name : target_names()) {
                const af::array& desired_seq = ys_groups.at(2).at(target_name);
                for (long time = 0; time < validation_size - n_steps_ahead_; ++time) {
                    af::array predicted_subseq = af::constant(af::NaN, n_steps_ahead_);
                    for (long i = 1; i <= n_steps_ahead_; ++i) {
                        std::string shifted_name = target_name + "-" + std::to_string(i);
                        predicted_subseq(i - 1) = output.at(shifted_name)(time);
                    }
                    predicted.emplace_back(target_name, std::move(predicted_subseq));
                    desired.emplace_back(
                      target_name, desired_seq(af::seq(time + 1, time + n_steps_ahead_)));
                }

                std::string last_shift_name = target_name + "-" + std::to_string(n_steps_ahead_);
                last_shift_predicted.emplace(last_shift_name, output.at(last_shift_name));
                last_shift_desired.emplace(
                  last_shift_name, af::shift(desired_seq, -n_steps_ahead_)(valid_selector));
            }
            error = multi_error_fnc(predicted, desired);
            std::cout << "Epoch " << epoch << " error " << error << "\n"
                      << "Error of the last target "
                      << error_fnc(last_shift_predicted, last_shift_desired) << std::endl;
        }
        return error;
    }

    virtual const std::set<std::string>& target_names() const = 0;
};

/// Base class for echo state network benchmarking tasks.
class benchmark_set : public benchmark_set_base {
protected:
    /// Generate the training inputs and outputs of dimensions [n_ins, len] and [n_outs, len].
    virtual std::tuple<data_map, data_map>
    generate_data(long len, af::dtype dtype, std::mt19937& prng) const = 0;

public:
    using benchmark_set_base::benchmark_set_base;

    double evaluate(net_base& net, std::mt19937& prng) const override
    {
        long len = rg::accumulate(split_sizes_, 0L);
        data_map xs, ys;
        std::tie(xs, ys) = generate_data(len, net.state().type(), prng);
        assert(xs.keys() == input_names());
        assert(xs.keys() == net.input_names());
        assert(xs.length() >= len);
        assert(ys.keys() == output_names());
        assert(ys.keys() == net.output_names());
        assert(ys.length() >= len);
        // use the input transform
        // split both input and output to init, train, test
        std::vector<data_map> xs_groups = xs.split(split_sizes_);
        std::vector<data_map> ys_groups = ys.split(split_sizes_);
        double error = std::numeric_limits<double>::quiet_NaN();
        for (long epoch = 0; epoch < n_epochs_; ++epoch) {
            // initialize the network using the initial sequence
            net.event("init-start");
            net.random_noise(true);
            net.feed(
              {.input = xs_groups.at(0),
               .feedback = ys_groups.at(0),
               .desired = ys_groups.at(0),
               .input_transform = input_transform_fn()});
            // train the network on the training sequence
            // teacher-force the first epoch, but not the others
            net.event("train-start");
            if (epoch == 0)
                net.train(
                  {.input = xs_groups.at(1),
                   .feedback = ys_groups.at(1),
                   .desired = ys_groups.at(1),
                   .input_transform = input_transform_fn()});
            else
                net.train(
                  {.input = xs_groups.at(1),
                   .feedback = {},
                   .desired = ys_groups.at(1),
                   .input_transform = input_transform_fn()});
            net.random_noise(false);
            net.clear_feedback();
            // evaluate the performance of the network on the validation sequence
            // note no teacher forcing
            net.event("validation-start");
            feed_result_t feed_result = net.feed(
              {.input = xs_groups.at(2),
               .feedback = {},
               .desired = ys_groups.at(2),
               .input_transform = input_transform_fn()});
            data_map predicted{net.output_names(), feed_result.outputs};
            error = error_fnc(predicted, ys_groups.at(2));
            std::cout << "Epoch " << epoch << " " << error << std::endl;
        }
        return error;
    }
};

/// Evaluate the performance of the given network on sequence prediction.
///
/// The network is trained on `train-steps` steps and simulated
/// for `n-steps-ahead`. Then it is trained on one more item and again
/// validated for `n-steps-ahead` until the end of `valid-steps` steps.
/// The error is calculated from all the validation steps.
class loop_benchmark_set : public benchmark_set_base {
protected:
    long n_steps_ahead_;
    long validation_stride_;

    /// Generate the training inputs and outputs of dimensions [n_ins, len] and [n_outs, len].
    virtual std::tuple<data_map, data_map>
    generate_data(af::dtype dtype, std::mt19937& prng) const = 0;

public:
    /// \param config The configuration parameters.
    ///        bench.split_sizes: The split sizes [init, train, n-steps-ahead].
    loop_benchmark_set(po::variables_map config)
      : benchmark_set_base{std::move(config)}
      , n_steps_ahead_{config_.at("bench.n-steps-ahead").as<long>()}
      , validation_stride_{config_.at("bench.validation-stride").as<long>()}
    {
        split_sizes_ = {
          config_.at("bench.init-steps").as<long>(),   //
          config_.at("bench.train-steps").as<long>(),  //
          config_.at("bench.valid-steps").as<long>()};
        assert(n_steps_ahead_ <= split_sizes_.at(2));
    }

    double evaluate(net_base& net, std::mt19937& prng) const override
    {
        assert((long)split_sizes_.size() == 3);
        // retrieve the data
        data_map xs, ys;
        std::tie(xs, ys) = generate_data(net.state().type(), prng);
        assert(xs.keys() == net.input_names());
        assert(ys.keys() == net.output_names());
        assert(xs.length() == ys.length());
        if (xs.length() < rg::accumulate(split_sizes_, 0L))
            throw std::runtime_error{
              "Not enough data in the dataset for the given split sizes. Data have "
              + std::to_string(xs.length()) + " and split sizes "
              + std::to_string(rg::accumulate(split_sizes_, 0L)) + "."};
        // split the sequences into groups
        std::vector<data_map> xs_groups = xs.split(split_sizes_);
        std::vector<data_map> ys_groups = ys.split(split_sizes_);
        double error = std::numeric_limits<double>::quiet_NaN();
        for (long epoch = 0; epoch < n_epochs_; ++epoch) {
            // initialize the network using the initial sequence
            net.event("init-start");
            net.random_noise(true);
            net.feed(
              {.input = xs_groups.at(0),
               .feedback = ys_groups.at(0),
               .desired = ys_groups.at(0),
               .input_transform = input_transform_fn()});
            // train the network on the training sequence with teacher forcing
            train_result_t train_result = [&]() {
                net.event("train-start");
                if (epoch == 0)
                    return net.train(
                      {.input = xs_groups.at(1),
                       .feedback = ys_groups.at(1),
                       .desired = ys_groups.at(1),
                       .input_transform = input_transform_fn()});
                train_result_t train_result = net.train(
                  {.input = xs_groups.at(1).probably_nan(output_names(), .5 * epoch / n_epochs_),
                   .feedback = {},
                   .desired = ys_groups.at(1),
                   .input_transform = input_transform_fn()});
                // give the network an unimpaired input of length `init-steps` after training
                // epoch>0 with probabilistically skipped values
                net.feed(
                  {.input = xs_groups.at(1).tail(split_sizes_.at(0)),
                   .feedback = ys_groups.at(1).tail(split_sizes_.at(0)),
                   .desired = ys_groups.at(1).tail(split_sizes_.at(0)),
                   .input_transform = input_transform_fn()});
                return train_result;
            }();
            net.random_noise(false);
            net.clear_feedback();
            {
                // Print train mse error
                af::array train_prediction =
                  af::matmul(train_result.predictors, train_result.output_w.T());
                double err = af_utils::mse<double>(train_prediction.T(), ys_groups.at(1).data());
                std::cout << "Train MSE error: " << err << std::endl;
            }
            // evaluate the performance of the network on all continuous intervals of the validation
            // sequence of length n_steps_ahead_ (except the last such interval)
            long n_validations =
              (xs_groups.at(2).length() - n_steps_ahead_ + validation_stride_ - 1)
              / validation_stride_;
            std::vector<data_map> all_predicted;
            std::vector<data_map> all_desired;
            long i;
            // the last step in the sequence has an unknown desired value, so we skip the
            // last sequence of n_steps_ahead_ (i.e., < instead of <=)
            for (i = 0; i < xs_groups.at(2).length() - n_steps_ahead_; i += validation_stride_) {
                assert(i / validation_stride_ < n_validations);
                // feed the network with the validation data before the validation subsequence
                if (i > 0) {
                    data_map input = xs_groups.at(2).select(af::seq(i - validation_stride_, i - 1));
                    data_map desired =
                      ys_groups.at(2).select(af::seq(i - validation_stride_, i - 1));
                    net.event("feed-extra");
                    net.feed(
                      {.input = input,
                       .feedback = desired,
                       .desired = desired,
                       .input_transform = input_transform_fn()});
                }
                // create a copy of the network before the validation so that we can simply
                // continue training of the original net in the next iteration
                std::unique_ptr<net_base> net_copy = net.clone();
                // evaluate the performance of the network on the validation subsequence
                data_map desired = ys_groups.at(2).select(af::seq(i, i + n_steps_ahead_ - 1));
                data_map loop_input = xs_groups.at(2)
                                        .select(af::seq(i, i + n_steps_ahead_ - 1))
                                        .filter(persistent_input_names());
                net_copy->event("validation-start");
                af::array raw_predicted = net_copy
                                            ->feed(
                                              {.input = loop_input,
                                               .feedback = {},
                                               .desired = desired,
                                               .input_transform = input_transform_fn()})
                                            .outputs;
                data_map predicted{output_names(), std::move(raw_predicted)};
                {
                    double err = af_utils::mse<double>(predicted.data(), desired.data());
                    std::cout << "Valid MSE error: " << err << std::endl;
                }
                // extract the targets
                all_predicted.push_back(predicted.filter(target_names()));
                all_desired.push_back(desired.filter(target_names()));
            }
            assert(i / validation_stride_ == n_validations);
            error = multi_error_fnc(all_predicted, all_desired);
            std::cout << "Epoch " << epoch << " " << error << std::endl;
        }
        return error;
    }

    virtual const std::set<std::string>& persistent_input_names() const = 0;

    virtual const std::set<std::string>& target_names() const = 0;
};

/// The parameters are from Reservoir Topology in Deep Echo State Networks [2019]
/// by Gallicchio and Micheli.
class narma10_benchmark_set : public benchmark_set {
protected:
    long tau_;
    std::set<std::string> input_names_{"xs"};
    std::set<std::string> output_names_{"ys"};

    std::tuple<data_map, data_map>
    generate_data(long len, af::dtype dtype, std::mt19937& prng) const override
    {
        // NARMA10 can diverge, let's regenerate until it all fits in [-1, 1].
        af::array xs, ys;
        af::randomEngine af_prng{AF_RANDOM_ENGINE_DEFAULT, prng()};
        do {
            xs = af::randu({len}, dtype, af_prng) * 0.5;
            ys = narma(xs, 10, tau_, {0.3, 0.05, 1.5, 0.1});
        } while (af::anyTrue<bool>(af::abs(ys) > 1.0));
        return {{"xs", xs}, {"ys", ys}};
    }

public:
    narma10_benchmark_set(po::variables_map config)
      : benchmark_set{std::move(config)}, tau_{config_.at("bench.narma-tau").as<long>()}
    {
    }

    const std::set<std::string>& input_names() const override
    {
        return input_names_;
    }

    const std::set<std::string>& output_names() const override
    {
        return output_names_;
    }
};

class narma30_benchmark_set : public benchmark_set {
protected:
    long tau_;
    std::set<std::string> input_names_{"xs"};
    std::set<std::string> output_names_{"ys"};

    std::tuple<data_map, data_map>
    generate_data(long len, af::dtype dtype, std::mt19937& prng) const override
    {
        // NARMA10 can diverge, not sure about NARMA30, let's rather check.
        af::array xs, ys;
        af::randomEngine af_prng{AF_RANDOM_ENGINE_DEFAULT, prng()};
        do {
            xs = af::randu({len}, dtype, af_prng) * 0.5;
            ys = narma(xs, 30, tau_, {0.2, 0.004, 1.5, 0.001});
        } while (af::anyTrue<bool>(af::abs(ys) > 1.0));
        return {{"xs", xs}, {"ys", ys}};
    }

public:
    narma30_benchmark_set(po::variables_map config)
      : benchmark_set{std::move(config)}, tau_{config_.at("bench.narma-tau").as<long>()}
    {
    }

    const std::set<std::string>& input_names() const override
    {
        return input_names_;
    }

    const std::set<std::string>& output_names() const override
    {
        return output_names_;
    }
};

class memory_capacity_benchmark_set : public benchmark_set {
protected:
    long history_;
    std::set<std::string> input_names_;
    std::set<std::string> output_names_;

    std::tuple<data_map, data_map>
    generate_data(long len, af::dtype dtype, std::mt19937& prng) const override
    {
        af::randomEngine af_prng{AF_RANDOM_ENGINE_DEFAULT, prng()};
        af::array xs = af::randu({len}, dtype, af_prng) * 2. - 1;
        af::array ys = memory_matrix(xs, history_);
        return {{"xs", xs}, {output_names_, ys}};
    }

    /// The `memory capacity` measure.
    ///
    /// \param predicted The predicted memory matrix.
    /// \param desired The gold memory matrix.
    /// \returns The average squared covariance of the corresponding columns.
    double memory_capacity(const data_map& predicted, const data_map& desired) const
    {
        assert(predicted.keys() == desired.keys());
        assert(predicted.keys() == output_names_);
        return af::sum<double>(
          af_utils::square(af_utils::cov(predicted.data(), desired.data(), 1)));
    }

    double error_fnc(const data_map& predicted, const data_map& desired) const override
    {
        if (error_measure_ == "memory-capacity") return memory_capacity(predicted, desired);
        return benchmark_set::error_fnc(predicted, desired);
    }

public:
    memory_capacity_benchmark_set(po::variables_map config)
      : benchmark_set{std::move(config)}, history_{config_.at("bench.memory-history").as<long>()}
    {
        if (history_ < 1)
            throw std::invalid_argument(
              "Memory history is too low (" + std::to_string(history_) + ").");
        input_names_ = {"xs"};
        for (long i = 0; i < history_; ++i) output_names_.insert("ys-" + std::to_string(i));
    }

    const std::set<std::string>& input_names() const override
    {
        return input_names_;
    }

    const std::set<std::string>& output_names() const override
    {
        return output_names_;
    }
};

/// Evaluate the performance of the given network on the memory mean squared error task for
/// a single fixed delay.
class memory_single_benchmark_set : public benchmark_set {
protected:
    long history_;
    std::set<std::string> input_names_{"xs"};
    std::set<std::string> output_names_{"ys"};

    std::tuple<data_map, data_map>
    generate_data(long len, af::dtype dtype, std::mt19937& prng) const override
    {
        af::randomEngine af_prng{AF_RANDOM_ENGINE_DEFAULT, prng()};
        af::array xs = af::randu({len}, dtype, af_prng) * 2. - 1;
        af::array ys = af::shift(xs, history_);
        return {{"xs", xs}, {"ys", ys}};
    }

public:
    memory_single_benchmark_set(po::variables_map config)
      : benchmark_set{std::move(config)}, history_{config_.at("bench.memory-history").as<long>()}
    {
        if (history_ < 1)
            throw std::invalid_argument(
              "Memory history is too low (" + std::to_string(history_) + ").");
    }

    const std::set<std::string>& input_names() const override
    {
        return input_names_;
    }

    const std::set<std::string>& output_names() const override
    {
        return output_names_;
    }
};

class mackey_glass_benchmark_set : public benchmark_set {
protected:
    long tau_;
    double delta_;
    std::set<std::string> input_names_{"xs"};
    std::set<std::string> output_names_{"ys"};

    data_map input_transform(const data_map& xs) const override
    {
        if (xs.empty()) return {};
        return {xs.keys(), af::tanh(xs.data() - 1.)};
    }

    std::tuple<data_map, data_map>
    generate_data(long len, af::dtype dtype, std::mt19937& prng) const override
    {
        af::array mg = esn::mackey_glass(len + 1, tau_, delta_, dtype, prng);
        af::array xs = mg(af::seq(0, af::end - 1));
        af::array ys = mg(af::seq(1, af::end));
        return {{"xs", xs}, {"ys", ys}};
    }

public:
    mackey_glass_benchmark_set(po::variables_map config)
      : benchmark_set{std::move(config)}
      , tau_{config_.at("bench.mackey-glass-tau").as<long>()}
      , delta_{config_.at("bench.mackey-glass-delta").as<double>()}
    {
    }

    const std::set<std::string>& input_names() const override
    {
        return input_names_;
    }

    const std::set<std::string>& output_names() const override
    {
        return output_names_;
    }
};

class ett_loader {
protected:
    fs::path data_path_;
    std::vector<std::string> header_ = {"date", "HUFL", "HULL", "MUFL",
                                        "MULL", "LUFL", "LULL", "OT"};
    data_map data_;

    data_map ett_input_transform(const data_map& xs) const
    {
        if (xs.empty()) return {};
        // Avoid -1 and 1 to atanh by multiplying by 0.99.
        return {xs.keys(), af::tanh((xs.data() / 50. - 0.2)) * 0.99};
    }

public:
    ett_loader(const po::variables_map& config)
      : data_path_{config.at("bench.ett-data-path").as<std::string>()}
    {
    }

    void load_data(const fs::path& csv_path)
    {
        if (!fs::exists(data_path_)) throw std::runtime_error{"ETT data path does not exist."};
        fs::path csv = data_path_ / csv_path;
        if (!fs::exists(csv)) throw std::runtime_error{"ETT csv file does not exist."};
        std::ifstream in{csv};
        std::string line;
        std::getline(in, line);
        if (line != boost::join(header_, ",")) throw std::runtime_error{"Invalid header."};

        std::map<std::string, std::vector<double>> data;
        while (std::getline(in, line)) {
            std::vector<std::string> words;
            boost::split(words, line, boost::is_any_of(","));
            for (const auto& [col, value] : rgv::zip(header_, words))
                if (col == "date") {
                    std::tm tm = {};
                    std::stringstream ss{value};
                    ss >> std::get_time(&tm, "%Y-%m-%d %H:%M:%S");
                    data["date-mon"].push_back(
                      std::sin(
                        (tm.tm_mon * 31 * 24 + tm.tm_mday * 24 + tm.tm_hour) * 2. * M_PI
                        / (12 * 31 * 24))
                      * 10);
                    data["date-mday"].push_back(
                      std::sin((tm.tm_mday * 24 + tm.tm_hour) * 2. * M_PI / (31 * 24)) * 10);
                    data["date-wday"].push_back(
                      std::sin((tm.tm_wday * 24 + tm.tm_hour) * 2. * M_PI / (7 * 24)) * 10);
                    data["date-hour"].push_back(std::sin(tm.tm_hour * 2. * M_PI / 24) * 10);
                    data["date-min"].push_back(std::sin(tm.tm_min * 2. * M_PI / 60) * 10);
                } else
                    data[col].push_back(std::stod(value));
        };

        long long n_features = header_.size() - 1;
        long long n_points = data.at("OT").size();

        std::map<std::string, af::array> array_data;
        for (const auto& [key, values] : data) array_data[key] = af_utils::to_array(values);
        data_ = array_data;

        std::cout << "Loaded ETT dataset with " << n_features << " features and " << n_points
                  << " points.\n";
    }
};

class etth_loader : public ett_loader {
protected:
    int variant_;
    std::string set_type_;  // train/valid/test

    const data_map& get_dataset(af::dtype dtype, std::mt19937& prng) const
    {
        data_map xs;
        if (set_type_ == "train") return train_data_;
        if (set_type_ == "valid") return valid_data_;
        if (set_type_ == "train-valid") return train_valid_data_;
        if (set_type_ == "test") return test_data_;
        throw std::runtime_error{"Unknown dataset."};
    }

    data_map train_data_;
    data_map valid_data_;
    data_map train_valid_data_;
    data_map test_data_;

public:
    etth_loader(po::variables_map config)
      : ett_loader{config}
      , variant_{config.at("bench.etth-variant").as<int>()}
      , set_type_{config.at("bench.ett-set-type").as<std::string>()}
    {
        load_data("ETT-small/ETTh" + std::to_string(variant_) + ".csv");

        train_data_ = data_.select(af::seq(0, 12 * 30 * 24));
        std::cout << "ETT train has " << train_data_.length() << " points.\n";

        valid_data_ = data_.select(af::seq(12 * 30 * 24, (12 + 4) * 30 * 24));
        std::cout << "ETT valid has " << valid_data_.length() << " points.\n";

        train_valid_data_ = data_.select(af::seq(0, (12 + 4) * 30 * 24));
        std::cout << "ETT train-valid has " << train_valid_data_.length() << " points.\n";

        test_data_ = data_.select(af::seq((12 + 4) * 30 * 24, (12 + 4 + 4) * 30 * 24));
        std::cout << "ETT test has " << test_data_.length() << " points.\n";

        std::cout << "Naive 1-step ahead prediction valid MSE error is "
                  << af_utils::mse<double>(
                       valid_data_.at("OT"), af::shift(valid_data_.at("OT"), -1))
                  << std::endl;
    }
};

class ettm_loader : public ett_loader {
protected:
    int variant_;
    std::string set_type_;  // train/valid/test

    const data_map& get_dataset(af::dtype dtype, std::mt19937& prng) const
    {
        data_map xs;
        if (set_type_ == "train") return train_data_;
        if (set_type_ == "valid") return valid_data_;
        if (set_type_ == "train-valid") return train_valid_data_;
        if (set_type_ == "test") return test_data_;
        throw std::runtime_error{"Unknown dataset."};
    }

    data_map train_data_;
    data_map valid_data_;
    data_map train_valid_data_;
    data_map test_data_;

public:
    ettm_loader(po::variables_map config)
      : ett_loader{config}
      , variant_{config.at("bench.etth-variant").as<int>()}
      , set_type_{config.at("bench.ett-set-type").as<std::string>()}
    {
        load_data("ETT-small/ETTm" + std::to_string(variant_) + ".csv");

        train_data_ = data_.select(af::seq(0, 12 * 30 * 24 * 4));
        std::cout << "ETT train has " << train_data_.length() << " points.\n";

        valid_data_ = data_.select(af::seq(12 * 30 * 24 * 4, (12 + 4) * 30 * 24 * 4));
        std::cout << "ETT valid has " << valid_data_.length() << " points.\n";

        train_valid_data_ = data_.select(af::seq(0, (12 + 4) * 30 * 24 * 4));
        std::cout << "ETT train-valid has " << train_valid_data_.length() << " points.\n";

        test_data_ = data_.select(af::seq((12 + 4) * 30 * 24 * 4, (12 + 4 + 4) * 30 * 24 * 4));
        std::cout << "ETT test has " << test_data_.length() << " points.\n";

        std::cout << "Naive 1-step ahead prediction valid MSE error is "
                  << af_utils::mse<double>(
                       valid_data_.at("OT"), af::shift(valid_data_.at("OT"), -1))
                  << std::endl;
    }
};

class etth_loop_benchmark_set : public loop_benchmark_set, public etth_loader {
protected:
    std::set<std::string> persistent_input_names_{
      "date-mon", "date-mday", "date-wday", "date-hour"};
    std::set<std::string> input_names_{"date-mon", "date-mday", "date-wday", "date-hour",
                                       "HUFL",     "HULL",      "MUFL",      "MULL",
                                       "LUFL",     "LULL",      "OT"};
    std::set<std::string> output_names_{"HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"};
    std::set<std::string> target_names_{"OT"};

    data_map input_transform(const data_map& xs) const override
    {
        return ett_input_transform(xs);
    }

    std::tuple<data_map, data_map> generate_data(af::dtype dtype, std::mt19937& prng) const override
    {
        data_map dataset = get_dataset(dtype, prng);
        data_map xs = dataset.filter(input_names());
        data_map ys = dataset.filter(output_names()).shift(-1);
        return {std::move(xs), std::move(ys)};
    }

public:
    etth_loop_benchmark_set(po::variables_map config)
      : loop_benchmark_set{std::move(config)}, etth_loader{config_}
    {
    }

    const std::set<std::string>& persistent_input_names() const override
    {
        return persistent_input_names_;
    }

    const std::set<std::string>& input_names() const override
    {
        return input_names_;
    }

    const std::set<std::string>& output_names() const override
    {
        return output_names_;
    }

    const std::set<std::string>& target_names() const override
    {
        return target_names_;
    }

    bool constant_data() const override
    {
        return true;
    }
};

class ettm_loop_benchmark_set : public loop_benchmark_set, public ettm_loader {
protected:
    std::set<std::string> persistent_input_names_{
      "date-mon", "date-mday", "date-wday", "date-hour", "date-min"};
    std::set<std::string> input_names_{"date-mon", "date-mday", "date-wday", "date-hour",
                                       "date-min", "HUFL",      "HULL",      "MUFL",
                                       "MULL",     "LUFL",      "LULL",      "OT"};
    std::set<std::string> output_names_{"HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"};
    std::set<std::string> target_names_{"OT"};

    data_map input_transform(const data_map& xs) const override
    {
        return ett_input_transform(xs);
    }

    std::tuple<data_map, data_map> generate_data(af::dtype dtype, std::mt19937& prng) const override
    {
        data_map dataset = get_dataset(dtype, prng);
        data_map xs = dataset.filter(input_names());
        data_map ys = dataset.filter(output_names()).shift(-1);
        return {std::move(xs), std::move(ys)};
    }

public:
    ettm_loop_benchmark_set(po::variables_map config)
      : loop_benchmark_set{std::move(config)}, ettm_loader{config_}
    {
    }

    const std::set<std::string>& persistent_input_names() const override
    {
        return persistent_input_names_;
    }

    const std::set<std::string>& input_names() const override
    {
        return input_names_;
    }

    const std::set<std::string>& output_names() const override
    {
        return output_names_;
    }

    const std::set<std::string>& target_names() const override
    {
        return target_names_;
    }

    bool constant_data() const override
    {
        return true;
    }
};

class etth_markov_benchmark_set : public markov_benchmark_set, public etth_loader {
protected:
    std::set<std::string> input_names_{"date-mon", "date-mday", "date-wday", "date-hour",
                                       "HUFL",     "HULL",      "MUFL",      "MULL",
                                       "LUFL",     "LULL",      "OT"};
    std::set<std::string> target_names_ = {"OT"};
    std::set<std::string> output_names_ = make_shifted_seq_names(target_names_, n_steps_ahead_);

    data_map input_transform(const data_map& xs) const override
    {
        return ett_input_transform(xs);
    }

    std::tuple<data_map, data_map>
    generate_data(long len, af::dtype dtype, std::mt19937& prng) const override
    {
        data_map xs = get_dataset(dtype, prng);
        data_map ys = xs.filter(target_names());
        return {std::move(xs), std::move(ys)};
    }

public:
    etth_markov_benchmark_set(po::variables_map config)
      : markov_benchmark_set{std::move(config)}, etth_loader{config_}
    {
    }

    const std::set<std::string>& input_names() const override
    {
        return input_names_;
    }

    const std::set<std::string>& output_names() const override
    {
        return output_names_;
    }

    const std::set<std::string>& target_names() const override
    {
        return target_names_;
    }

    bool constant_data() const override
    {
        return true;
    }
};

class lyapunov_benchmark_set : public benchmark_set_base {
protected:
    long init_len_;
    long seq_len_;
    double d0_;
    long n_retries_;
    std::set<std::string> input_names_;
    std::set<std::string> output_names_;

    double lyapunov_trial_(const net_base& net_, double d0, af::randomEngine& af_prng) const
    {
        // create a copy of the network to work with
        std::unique_ptr<net_base> net = net_.clone();
        assert(net->input_names() == input_names_);
        assert(net->output_names() == output_names_);
        auto dtype = net->state().type();
        // pass initial transitions by feeding a random sequence
        af::array init_xs =
          af::randu({(dim_t)net->input_names().size(), init_len_}, dtype, af_prng) * 0.5;
        net->feed(
          {.input = {"xs", init_xs},
           .feedback = {},
           .desired = {},
           .input_transform = input_transform_fn()});
        // build the sequence to be analyzed
        af::array xs =
          af::randu({(dim_t)net->input_names().size(), seq_len_}, dtype, af_prng) * 0.5;
        // make a soon-to-be perturbed clone of the original net
        net->random_noise(false);
        std::unique_ptr<net_base> net_pert = net->clone();
        // allocate the vector of distances between the states (for each time step)
        af::array dists_time = af::array(seq_len_, dtype);
        // for each time step
        for (int t = 0; t < seq_len_; ++t) {
            // perform a single step on the cloned net
            net->step({"xs", xs(af::span, t)}, {}, {}, input_transform_fn());
            // perform a single step on the perturbed net (and perturb input in time 0)
            if (t == 0)
                net_pert->step({"xs", xs(af::span, t) + d0}, {}, {}, input_transform_fn());
            else
                net_pert->step({"xs", xs(af::span, t)}, {}, {}, input_transform_fn());
            // calculate the distance between net and net_pert states
            double norm = af::norm(net->state() - net_pert->state(), AF_NORM_VECTOR_2);
            dists_time(t) = norm;
            // normalize the net_pert state back to the distance of d0 from net
            double norm_coef = d0 / norm;
            net_pert->state(net->state() + (net_pert->state() - net->state()) * norm_coef);
        }
        // calculate the lyapunov exponents for each step of the current sequence
        // if dist == 0 in some moment, then log is NaN, and it may represents an
        // ordered state or that the network state hits the precision of the
        // underlying data type; if this happens to you, try to increase d0
        af::array lyaps_time = af::log(dists_time / d0);
        // calculate the average lyapunov exponent of the current sequence
        // lyap exp can have very low and very high values, rather clip it to [-1, 1]
        return af::mean<double>(af_utils::clip(lyaps_time, -1., 1.));
    }

public:
    /// Construct the lyapunov exponent benchmark for echo state networks.
    ///
    /// If you encounter NaN values, try to increase the value of d0. The reason
    /// may be that d0 * input_w hit the maximum precision of the underlying
    /// data type try to change d0 so that d0 * input_w is e.g., 10^-12.
    ///
    /// \param config
    ///         bench.init-steps: The length of the initial sequence used to pass transitions.
    ///         bench.valid-steps: The length of the sequence on which is the exponent calculated.
    /// \param d0 The size of the perturbation. If a NaN is encountered, d0 is multiplied
    ///           by 1e2 and the trial is retried.
    /// \param n_retries The maximum number of retries if NaN is encountered (set -1 for infinity).
    lyapunov_benchmark_set(
      po::variables_map config,
      std::set<std::string> input_names = {"xs"},
      std::set<std::string> output_names = {"ys"},
      double d0 = 1e-12,
      long n_retries = 8)
      : benchmark_set_base{std::move(config)}
      , init_len_{config_.at("bench.init-steps").as<long>()}
      , seq_len_{config_.at("bench.valid-steps").as<long>()}
      , d0_{d0}
      , n_retries_{n_retries}
      , input_names_{std::move(input_names)}
      , output_names_{std::move(output_names)}
    {
    }

    double evaluate(net_base& net, std::mt19937& prng) const override
    {
        af::randomEngine af_prng{AF_RANDOM_ENGINE_DEFAULT, prng()};
        double d0 = d0_;
        // if the lyapunov exponent is NaN, increase d0 and try over
        for (int retry = 0; retry < n_retries_; ++retry) {
            double lyap = lyapunov_trial_(net, d0, af_prng);
            if (!std::isnan(lyap)) return lyap;
            d0 *= 1e2;
        }
        return std::numeric_limits<double>::quiet_NaN();
    }

    const std::set<std::string>& input_names() const override
    {
        return input_names_;
    }

    const std::set<std::string>& output_names() const override
    {
        return output_names_;
    }
};

/// A dummy benchmark set flipping the input between 1 and -1.
class semaphore_benchmark_set : public benchmark_set_base {
protected:
    long period_;
    std::set<std::string> input_names_{"xs"};
    std::set<std::string> output_names_{"ys"};

public:
    semaphore_benchmark_set(
      po::variables_map config,
      long period = 100,
      std::set<std::string> input_names = {"xs"},
      std::set<std::string> output_names = {"ys"})
      : benchmark_set_base{std::move(config)}
      , period_{config_.at("bench.period").as<long>()}
      , input_names_{std::move(input_names)}
      , output_names_{std::move(output_names)}
    {
    }

    double evaluate(net_base& net, std::mt19937&) const override
    {
        assert(net.input_names() == input_names_ && net.output_names() == output_names_);
        for (long time = 0;; ++time) {
            af::array in = af::constant(2. * (time / period_ % 2) - 1, 1);
            net.step({"xs", -in}, {"ys", -in}, {"ys", -in}, input_transform_fn());
        }
    }

    const std::set<std::string>& input_names() const override
    {
        return input_names_;
    }

    const std::set<std::string>& output_names() const override
    {
        return output_names_;
    }
};

inline std::unique_ptr<benchmark_set_base> make_benchmark(const po::variables_map& args)
{
    if (args.at("gen.benchmark-set").as<std::string>() == "narma10") {
        return std::make_unique<narma10_benchmark_set>(args);
    }
    if (args.at("gen.benchmark-set").as<std::string>() == "narma30") {
        return std::make_unique<narma30_benchmark_set>(args);
    }
    if (args.at("gen.benchmark-set").as<std::string>() == "memory-capacity") {
        return std::make_unique<memory_capacity_benchmark_set>(args);
    }
    if (args.at("gen.benchmark-set").as<std::string>() == "memory-single") {
        return std::make_unique<memory_single_benchmark_set>(args);
    }
    if (args.at("gen.benchmark-set").as<std::string>() == "mackey-glass") {
        return std::make_unique<mackey_glass_benchmark_set>(args);
    }
    if (args.at("gen.benchmark-set").as<std::string>() == "lyapunov") {
        return std::make_unique<lyapunov_benchmark_set>(args);
    }
    if (args.at("gen.benchmark-set").as<std::string>() == "semaphore") {
        return std::make_unique<semaphore_benchmark_set>(args);
    }
    if (args.at("gen.benchmark-set").as<std::string>() == "etth-loop") {
        return std::make_unique<etth_loop_benchmark_set>(args);
    }
    if (args.at("gen.benchmark-set").as<std::string>() == "ettm-loop") {
        return std::make_unique<ettm_loop_benchmark_set>(args);
    }
    if (args.at("gen.benchmark-set").as<std::string>() == "etth-markov") {
        return std::make_unique<etth_markov_benchmark_set>(args);
    }
    throw std::runtime_error{
      "Unknown benchmark \"" + args.at("gen.benchmark-set").as<std::string>() + "\"."};
}

inline po::options_description benchmark_arg_description()
{
    // TODO move to benchmarks
    po::options_description benchmark_arg_desc{"Benchmark options"};
    benchmark_arg_desc.add_options()                                                             //
      ("bench.memory-history", po::value<long>()->default_value(0),                              //
       "The length of the memory to be evaluated.")                                              //
      ("bench.n-steps-ahead", po::value<long>()->default_value(84),                              //
       "The length of the valid sequence in sequence prediction benchmark.")                     //
      ("bench.validation-stride", po::value<long>()->default_value(1),                           //
       "Stride of validation subsequences (of length n-steps-ahead).")                           //
      ("bench.mackey-glass-tau", po::value<long>()->default_value(30),                           //
       "The length of the memory to be evaluated.")                                              //
      ("bench.mackey-glass-delta", po::value<double>()->default_value(0.1),                      //
       "The time delta (and subsampling) for mackey glass equations.")                           //
      ("bench.narma-tau", po::value<long>()->default_value(1),                                   //
       "The time lag for narma series.")                                                         //
      ("bench.error-measure", po::value<std::string>()->default_value("mse"),                    //
       "The error function to be used. One of mse, nmse, nrmse.")                                //
      ("bench.n-trials", po::value<long>()->default_value(1),                                    //
       "The number of repeats of the [teacher-force, valid] step in the "                        //
       "sequence prediction benchmark.")                                                         //
      ("bench.n-epochs", po::value<long>()->default_value(1),                                    //
       "The number of retrainings of the network. Only the first epoch is teacher-forced.")      //
      ("bench.init-steps", po::value<long>()->default_value(1000),                               //
       "The number of training time steps.")                                                     //
      ("bench.train-steps", po::value<long>()->default_value(5000),                              //
       "The number of valid time steps.")                                                        //
      ("bench.valid-steps", po::value<long>()->default_value(1000),                              //
       "The number of test time steps.")                                                         //
      ("bench.period", po::value<long>()->default_value(100),                                    //
       "The period of flipping the semaphore sign.")                                             //
      ("bench.ett-data-path", po::value<std::string>()->default_value("third_party/ETDataset"),  //
       "Path to the ETT dataset.")                                                               //
      ("bench.etth-variant", po::value<int>()->default_value(1),                                 //
       "Variant of the ETTh dataset (1 or 2).")                                                  //
      ("bench.ett-set-type", po::value<std::string>()->default_value("train-valid"),             //
       "Part of the ETT dataset (train, valid, train-valid, test).")                             //
      ;
    return benchmark_arg_desc;
}

}  // namespace esn