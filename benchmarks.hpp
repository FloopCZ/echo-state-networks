#pragma once

// The benchmarks used in combination with \ref optimizer. //

#include "analysis.hpp"
#include "argument_utils.hpp"
#include "net.hpp"

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

    virtual af::array input_transform(const af::array& xs) const
    {
        return xs;
    }

    virtual af::array output_transform(const af::array& ys) const
    {
        return ys;
    }

    virtual double error_fnc(const af::array& predicted, const af::array& desired) const
    {
        assert(predicted.dims() == desired.dims());
        assert(desired.numdims() == 2);
        // assert(desired.dims(0) == n_outs());  // not true for loop benchmark set
        assert(desired.dims(1) == split_sizes_.at(2));
        if (error_measure_ == "mse") return af_utils::mse<double>(predicted, desired);
        if (error_measure_ == "nmse") return af_utils::nmse<double>(predicted, desired);
        if (error_measure_ == "nrmse") return af_utils::nrmse<double>(predicted, desired);
        throw std::invalid_argument("Unknown error measure `" + error_measure_ + "`.");
    }

public:
    benchmark_set_base(po::variables_map config)
      : config_{std::move(config)}
      , split_sizes_{
            config_.at("bench.init-steps").as<long>()
          , config_.at("bench.train-steps").as<long>()
          , config_.at("bench.valid-steps").as<long>()}
      , error_measure_{config_.at("bench.error-measure").as<std::string>()}
    {
    }

    virtual double evaluate(net_base& net, std::mt19937& prng) const = 0;
    virtual long n_ins() const = 0;
    virtual long n_outs() const = 0;
    virtual ~benchmark_set_base() = default;
};

/// Base class for echo state network benchmarking tasks.
class benchmark_set : public benchmark_set_base {
protected:
    /// Generate the training inputs and outputs of dimensions [n_ins, len] and [n_outs, len].
    virtual std::tuple<af::array, af::array>
    generate_data(long len, af::dtype dtype, std::mt19937& prng) const = 0;

public:
    using benchmark_set_base::benchmark_set_base;

    double evaluate(net_base& net, std::mt19937& prng) const override
    {
        long len = rg::accumulate(split_sizes_, 0L);
        auto [xs, ys] = generate_data(len, net.state().type(), prng);
        assert(xs.numdims() == 2);
        assert(xs.dims(0) == n_ins());
        assert(xs.dims(0) == net.n_ins());
        assert(xs.dims(1) == len);
        assert(ys.numdims() == 2);
        assert(ys.dims(0) == n_outs());
        assert(ys.dims(0) == net.n_outs());
        assert(ys.dims(1) == len);
        // use the input transform
        // split both input and output to init, train, test
        std::vector<af::array> xs_groups = split_data(xs, split_sizes_);
        std::vector<af::array> ys_groups = split_data(ys, split_sizes_);
        // initialize the network using the initial sequence
        net.feed(input_transform(xs_groups[0]), input_transform(ys_groups[0]));
        // train the network on the training sequence
        net.train(input_transform(xs_groups[1]), input_transform(ys_groups[1]));
        // evaluate the performance of the network on the validation sequence
        // note no teacher forcing
        af::array ys_predict = net.feed(input_transform(xs_groups[2])).outputs;
        return error_fnc(output_transform(ys_predict), ys_groups[2]);
    }
};

/// Evaluate the performance of the given network on sequence prediction.
///
/// The n_ins and n_outs has to be equal to one.
/// The network is simulated for len(xs_test) steps and the error is
/// calculated from the last output.
class seq_prediction_benchmark_set : public benchmark_set_base {
protected:
    long n_trials_;

    double error_fnc(const af::array& predicted, const af::array& desired) const override
    {
        assert(predicted.dims() == desired.dims());
        assert(desired.numdims() == 2);
        assert(desired.dims(0) == n_trials_);
        assert(desired.dims(1) == split_sizes_.at(3));  // bench.n-steps-ahead
        // variance is taken from the whole desired sequence
        af::array var = af::var(af::flat(desired), AF_VARIANCE_DEFAULT);
        assert(var.isscalar());
        // mse only from the last output
        af::array mse = af_utils::mse(predicted(af::span, af::end), desired(af::span, af::end), 0);
        assert(mse.isscalar());

        if (error_measure_ == "mse") return mse.scalar<double>();
        if (error_measure_ == "nmse") return (mse / var).scalar<double>();
        if (error_measure_ == "nrmse") return (af::sqrt(mse / var)).scalar<double>();
        throw std::invalid_argument("Unknown error measure `" + error_measure_ + "`.");
    };

    // Generate the testing data. This should generate a sequence of dimensions [n_ins, len].
    virtual af::array generate_data(long len, af::dtype dtype, std::mt19937& prng) const = 0;

public:
    /// \param config The configuration parameters.
    ///        bench.split_sizes: The split sizes [init, train, teacher-force, n-steps-ahead].
    ///        bench.n-trials: The number of repeats of the [teacher-force, n-steps-ahead].
    seq_prediction_benchmark_set(po::variables_map config)
      : benchmark_set_base{std::move(config)}, n_trials_{config_.at("bench.n-trials").as<long>()}
    {
        split_sizes_ = {
          config_.at("bench.init-steps").as<long>(),           //
          config_.at("bench.train-steps").as<long>(),          //
          config_.at("bench.teacher-force-steps").as<long>(),  //
          config_.at("bench.n-steps-ahead").as<long>()};
        // Repeat the [teacher-force, test] procedure n_trials times.
        for (long i = 1; i < n_trials_; ++i) {
            split_sizes_.push_back(split_sizes_.at(2));
            split_sizes_.push_back(split_sizes_.at(3));
        }
    }

    double evaluate(net_base& net, std::mt19937& prng) const override
    {
        assert(net.n_ins() == 1);
        assert(net.n_ins() == net.n_outs());
        assert((long)split_sizes_.size() == 2 + n_trials_ * 2);
        // generate the input sequence
        af::array xs = generate_data(rg::accumulate(split_sizes_, 1L), net.state().type(), prng);
        assert(xs.type() == net.state().type());
        assert(xs.numdims() == 2);
        assert(xs.dims(0) == net.n_ins());
        assert(xs.dims(1) == rg::accumulate(split_sizes_, 1L));
        // generate the desired sequence
        af::array ys = af::shift(xs, 0, -1);
        xs = xs(af::span, af::seq(0, af::end - 1));
        ys = ys(af::span, af::seq(0, af::end - 1));
        assert(xs.dims(1) == rg::accumulate(split_sizes_, 0L));
        assert(ys.dims(0) == net.n_outs());
        assert(ys.dims(1) == rg::accumulate(split_sizes_, 0L));
        // transform the input sequence
        // split the sequences into groups
        std::vector<af::array> xs_groups = split_data(xs, split_sizes_);
        std::vector<af::array> ys_groups = split_data(ys, split_sizes_);
        // initialize the network using the initial sequence
        net.feed(input_transform(xs_groups.at(0)), input_transform(ys_groups.at(0)));
        // train the network on the training sequence
        net.train(input_transform(xs_groups.at(1)), input_transform(ys_groups.at(1)));
        // the rest are pairs (teacher-force, test)
        af::array predicted = af::constant(af::NaN, n_trials_, split_sizes_.at(3), xs.type());
        af::array desired = af::constant(af::NaN, n_trials_, split_sizes_.at(3), xs.type());
        for (long trial = 0; trial < n_trials_; ++trial) {
            // teacher-force the tf sequence
            net.feed(
              input_transform(xs_groups.at(2 * trial + 2)),
              input_transform(ys_groups.at(2 * trial + 2)));
            // predict and evaluate, note no teacher forcing
            predicted(trial, af::span) =
              net.loop(split_sizes_.at(3), input_transform(ys_groups.at(2 * trial + 3))).outputs;
            desired(trial, af::span) = ys_groups.at(2 * trial + 3);
        }
        // invoke the error function
        return error_fnc(output_transform(predicted), desired);
    }
};

/// The parameters are from Reservoir Topology in Deep Echo State Networks [2019]
/// by Gallicchio and Micheli.
class narma10_benchmark_set : public benchmark_set {
protected:
    long tau_;

    std::tuple<af::array, af::array>
    generate_data(long len, af::dtype dtype, std::mt19937& prng) const override
    {
        // NARMA10 can diverge, let's regenerate until it all fits in [-1, 1].
        af::array xs, ys;
        af::randomEngine af_prng{AF_RANDOM_ENGINE_DEFAULT, prng()};
        do {
            xs = af::randu({len}, dtype, af_prng) * 0.5;
            ys = narma(xs, 10, tau_, {0.3, 0.05, 1.5, 0.1});
        } while (af::anyTrue<bool>(af::abs(ys) > 1.0));
        return {xs.T(), ys.T()};
    }

public:
    narma10_benchmark_set(po::variables_map config)
      : benchmark_set{std::move(config)}, tau_{config_.at("bench.narma-tau").as<long>()}
    {
    }

    long n_ins() const override
    {
        return 1;
    }

    long n_outs() const override
    {
        return 1;
    }
};

class narma30_benchmark_set : public benchmark_set {
protected:
    long tau_;

    std::tuple<af::array, af::array>
    generate_data(long len, af::dtype dtype, std::mt19937& prng) const override
    {
        // NARMA10 can diverge, not sure about NARMA30, let's rather check.
        af::array xs, ys;
        af::randomEngine af_prng{AF_RANDOM_ENGINE_DEFAULT, prng()};
        do {
            xs = af::randu({len}, dtype, af_prng) * 0.5;
            ys = narma(xs, 30, tau_, {0.2, 0.004, 1.5, 0.001});
        } while (af::anyTrue<bool>(af::abs(ys) > 1.0));
        return {xs.T(), ys.T()};
    }

public:
    narma30_benchmark_set(po::variables_map config)
      : benchmark_set{std::move(config)}, tau_{config_.at("bench.narma-tau").as<long>()}
    {
    }

    long n_ins() const override
    {
        return 1;
    }

    long n_outs() const override
    {
        return 1;
    }
};

class memory_capacity_benchmark_set : public benchmark_set {
protected:
    long history_;

    std::tuple<af::array, af::array>
    generate_data(long len, af::dtype dtype, std::mt19937& prng) const override
    {
        af::randomEngine af_prng{AF_RANDOM_ENGINE_DEFAULT, prng()};
        af::array xs = af::randu({len}, dtype, af_prng) * 2. - 1;
        af::array ys = memory_matrix(xs, history_);
        return {xs.T(), ys};
    }

    /// The `memory capacity` measure.
    ///
    /// \param predicted The predicted memory matrix.
    /// \param desired The gold memory matrix.
    /// \returns The average squared covariance of the corresponding columns.
    double memory_capacity(const af::array& predicted, const af::array& desired) const
    {
        assert(predicted.numdims() == 2 && desired.numdims() == 2);
        assert(predicted.dims(0) == history_);
        assert(desired.dims(0) == history_);
        assert(predicted.dims(1) == desired.dims(1));
        return af::sum<double>(af_utils::square(af_utils::cov(predicted, desired, 1)));
    }

    double error_fnc(const af::array& predicted, const af::array& desired) const override
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
    }

    long n_ins() const override
    {
        return 1;
    }

    long n_outs() const override
    {
        return history_;
    }
};

/// Evaluate the performance of the given network on the memory mean squared error task for
/// a single fixed delay.
class memory_single_benchmark_set : public benchmark_set {
protected:
    long history_;

    std::tuple<af::array, af::array>
    generate_data(long len, af::dtype dtype, std::mt19937& prng) const override
    {
        af::randomEngine af_prng{AF_RANDOM_ENGINE_DEFAULT, prng()};
        af::array xs = af::randu({len}, dtype, af_prng) * 2. - 1;
        af::array ys = af::shift(xs, history_);
        return {xs.T(), ys.T()};
    }

public:
    memory_single_benchmark_set(po::variables_map config)
      : benchmark_set{std::move(config)}, history_{config_.at("bench.memory-history").as<long>()}
    {
        if (history_ < 1)
            throw std::invalid_argument(
              "Memory history is too low (" + std::to_string(history_) + ").");
    }

    long n_ins() const override
    {
        return 1;
    }

    long n_outs() const override
    {
        return 1;
    }
};

class mackey_glass_benchmark_set : public benchmark_set {
protected:
    long tau_;
    double delta_;

    af::array input_transform(const af::array& xs) const override
    {
        return af::tanh(xs - 1.);
    }

    af::array output_transform(const af::array& ys) const override
    {
        return af::atanh(ys) + 1.;
    }

    std::tuple<af::array, af::array>
    generate_data(long len, af::dtype dtype, std::mt19937& prng) const override
    {
        af::array mg = esn::mackey_glass(len + 1, tau_, delta_, dtype, prng);
        af::array xs = mg(af::seq(0, af::end - 1));
        af::array ys = mg(af::seq(1, af::end));
        return {xs.T(), ys.T()};
    }

public:
    mackey_glass_benchmark_set(po::variables_map config)
      : benchmark_set{std::move(config)}
      , tau_{config_.at("bench.mackey-glass-tau").as<long>()}
      , delta_{config_.at("bench.mackey-glass-delta").as<double>()}
    {
    }

    long n_ins() const override
    {
        return 1;
    }
    long n_outs() const override
    {
        return 1;
    }
};

class mackey_glass_seq_benchmark_set : public seq_prediction_benchmark_set {
protected:
    long tau_;
    double delta_;

    af::array input_transform(const af::array& xs) const override
    {
        return af::tanh(xs - 1.);
    }

    af::array output_transform(const af::array& ys) const override
    {
        return af::atanh(ys) + 1.;
    }

    af::array generate_data(long len, af::dtype dtype, std::mt19937& prng) const override
    {
        return mackey_glass(len, tau_, delta_, dtype, prng).T();
    }

public:
    mackey_glass_seq_benchmark_set(po::variables_map config)
      : seq_prediction_benchmark_set{std::move(config)}
      , tau_{config_.at("bench.mackey-glass-tau").as<long>()}
      , delta_{config_.at("bench.mackey-glass-delta").as<double>()}
    {
        if (n_trials_ < 50) {
            std::cout << "WARNING: Mackey-Glass n_trials is low (" << n_trials_ << ")."
                      << std::endl;
        }
    }

    long n_ins() const override
    {
        return 1;
    }

    long n_outs() const override
    {
        return 1;
    }
};

class lyapunov_benchmark_set : public benchmark_set_base {
protected:
    long init_len_;
    long seq_len_;
    double d0_;
    long n_retries_;
    long n_ins_;
    long n_outs_;

    double lyapunov_trial_(const net_base& net_, double d0, af::randomEngine& af_prng) const
    {
        // create a copy of the network to work with
        std::unique_ptr<net_base> net = net_.clone();
        assert(net->n_ins() == net->n_outs());
        auto dtype = net->state().type();
        // pass initial transitions by feeding a random sequence
        af::array init_xs = af::randu({net->n_ins(), init_len_}, dtype, af_prng) * 0.5;
        net->feed(init_xs);
        // build the sequence to be analyzed
        af::array xs = af::randu({net->n_ins(), seq_len_}, dtype, af_prng) * 0.5;
        // make a soon-to-be perturbed clone of the original net
        net->random_noise(false);
        std::unique_ptr<net_base> net_pert = net->clone();
        // allocate the vector of distances between the states (for each time step)
        af::array dists_time = af::array(seq_len_, dtype);
        // for each time step
        for (int t = 0; t < seq_len_; ++t) {
            // perform a single step on the cloned net
            net->step(xs(af::span, t));
            // perform a single step on the perturbed net (and perturb input in time 0)
            if (t == 0)
                net_pert->step(xs(af::span, t) + d0);
            else
                net_pert->step(xs(af::span, t));
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
      long n_ins = 1,
      long n_outs = 1,
      double d0 = 1e-12,
      long n_retries = 8)
      : benchmark_set_base{std::move(config)}
      , init_len_{config_.at("bench.init-steps").as<long>()}
      , seq_len_{config_.at("bench.valid-steps").as<long>()}
      , d0_{d0}
      , n_retries_{n_retries}
      , n_ins_{n_ins}
      , n_outs_{n_outs}
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

    long n_ins() const override
    {
        return n_ins_;
    }

    long n_outs() const override
    {
        return n_outs_;
    }
};

/// A dummy benchmark set flipping the input between 1 and -1.
class semaphore_benchmark_set : public benchmark_set_base {
protected:
    long period_;
    long n_ins_;
    long n_outs_;

public:
    semaphore_benchmark_set(
      po::variables_map config, long period = 100, long n_ins = 1, long n_outs = 1)
      : benchmark_set_base{std::move(config)}
      , period_{config_.at("bench.period").as<long>()}
      , n_ins_{n_ins}
      , n_outs_{n_outs}
    {
    }

    double evaluate(net_base& net, std::mt19937&) const override
    {
        assert(net.n_ins() == n_ins_ && net.n_outs() == n_outs_);
        for (long time = 0;; ++time) {
            double in = 2 * (time / period_ % 2) - 1;
            net.step_constant(-in, -in);
        }
    }

    long n_ins() const override
    {
        return n_ins_;
    }

    long n_outs() const override
    {
        return n_outs_;
    }
};

std::unique_ptr<benchmark_set_base> make_benchmark(const po::variables_map& args)
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
    if (args.at("gen.benchmark-set").as<std::string>() == "mackey-glass-seq-prediction") {
        return std::make_unique<mackey_glass_seq_benchmark_set>(args);
    }
    if (args.at("gen.benchmark-set").as<std::string>() == "lyapunov") {
        return std::make_unique<lyapunov_benchmark_set>(args);
    }
    if (args.at("gen.benchmark-set").as<std::string>() == "semaphore") {
        return std::make_unique<semaphore_benchmark_set>(args);
    }
    throw std::runtime_error{
      "Unknown benchmark \"" + args.at("gen.benchmark-set").as<std::string>() + "\"."};
}

po::options_description benchmark_arg_description()
{
    // TODO move to benchmarks
    po::options_description benchmark_arg_desc{"Benchmark options"};
    benchmark_arg_desc.add_options()                                            //
      ("bench.memory-history", po::value<long>()->default_value(0),             //
       "The length of the memory to be evaluated.")                             //
      ("bench.n-steps-ahead", po::value<long>()->default_value(84),             //
       "The length of the valid sequence in sequence prediction benchmark.")    //
      ("bench.mackey-glass-tau", po::value<long>()->default_value(30),          //
       "The length of the memory to be evaluated.")                             //
      ("bench.mackey-glass-delta", po::value<double>()->default_value(0.1),     //
       "The time delta (and subsampling) for mackey glass equations.")          //
      ("bench.narma-tau", po::value<long>()->default_value(1),                  //
       "The time lag for narma series.")                                        //
      ("bench.error-measure", po::value<std::string>()->default_value("mse"),   //
       "The error function to be used. One of mse, nmse, nrmse.")               //
      ("bench.n-trials", po::value<long>()->default_value(1),                   //
       "The number of repeats of the [teacher-force, valid] step in the "       //
       "sequence prediction benchmark.")                                        //
      ("bench.init-steps", po::value<long>()->default_value(1000),              //
       "The number of training time steps.")                                    //
      ("bench.train-steps", po::value<long>()->default_value(5000),             //
       "The number of valid time steps.")                                       //
      ("bench.valid-steps", po::value<long>()->default_value(1000),             //
       "The number of test time steps.")                                        //
      ("bench.teacher-force-steps", po::value<long>()->default_value(1000),     //
       "The number of teacher-force steps in sequence prediction benchmarks.")  //
      ("bench.period", po::value<long>()->default_value(100),                   //
       "The period of flipping the semaphore sign.")                            //
      ;
    return benchmark_arg_desc;
}

}  // namespace esn