#pragma once

// Echo state network optimization header. //

#include "analysis.hpp"
#include "benchmark_results.hpp"
#include "benchmarks.hpp"
#include "lcnn.hpp"
#include "net.hpp"
#include "simple_esn.hpp"

#include <boost/program_options.hpp>
#include <execution>
#include <iostream>
#include <libcmaes/cmaes.h>
#include <limits>
#include <map>
#include <mutex>
#include <range/v3/all.hpp>

namespace esn {

namespace po = boost::program_options;
namespace rg = ranges;
namespace rgv = ranges::views;

/// Generic function optimizer.
template <typename EvaluationResult>
class optimizer {
protected:
    po::variables_map config_;
    using GenoPheno = cma::GenoPheno<cma::pwqBoundStrategy, cma::linScalingStrategy>;
    GenoPheno genopheno_;
    cma::CMAParameters<GenoPheno> cmaparams_;
    int n_evals_;
    std::string f_value_agg_;
    bool no_multithreading_;
    EvaluationResult best_evaluation_ = {.f_value = std::numeric_limits<double>::infinity()};
    std::mutex best_evaluation_mutex_;
    std::mt19937* prng_;

    // Reseed the random generator to a new state.
    void reseed()
    {
        std::seed_seq sseq{prng_->operator()(), prng_->operator()(), prng_->operator()()};
        prng_->seed(sseq);
    }

    /// Build the ProgressFunc for libcmaes.
    virtual cma::ProgressFunc<cma::CMAParameters<GenoPheno>, cma::CMASolutions>
    build_progress_func_()
    {
        return
          [this](const cma::CMAParameters<GenoPheno>& cmaparams, const cma::CMASolutions& cmasols) {
              std::cout << "Iteration: " << cmasols.niter() << '\n';
              std::cout << "Evaluations: " << cmasols.nevals() << '\n';
              std::cout << "Sigma: " << cmasols.sigma() << '\n';
              std::cout << "Min eigenvalue: " << cmasols.min_eigenv() << '\n';
              std::cout << "Max eigenvalue: " << cmasols.max_eigenv() << '\n';
              std::cout << "Elapsed time: " << cmasols.elapsed_last_iter() << '\n';
              std::cout << "Best candidate ";
              print_candidate(std::cout, cmasols.best_candidate()) << '\n';
              std::cout << std::endl;
              reseed();
              return 0;
          };
    }

    /// Build the FitFunc for libcmaes.
    cma::FitFunc build_fit_func_()
    {
        return [this](const double* p, int n) -> double {
            std::vector<double> params(p, p + n);
            std::mt19937 prng = *prng_;  // copy the generator for each f_value() call.
            return f_value(params, prng);
        };
    }

public:
    /// Dummy constructor.
    optimizer() = default;
    /// Construct from configuration.
    optimizer(po::variables_map config, std::mt19937& prng)
      : config_{std::move(config)}
      , n_evals_{config_.at("opt.n-evals").as<int>()}
      , f_value_agg_{config_.at("opt.f-value-agg").as<std::string>()}
      , no_multithreading_{config_.at("opt.no-multithreading").as<bool>()}
      , prng_{&prng}
    {
    }

    GenoPheno make_genopheno() const
    {
        std::vector<double> sigmas = param_sigmas();
        std::vector<double> lbounds = param_lbounds();
        std::vector<double> ubounds = param_ubounds();
        assert(sigmas.size() == lbounds.size());
        assert(lbounds.size() == ubounds.size());
        dVec scaling = dVec::Constant(sigmas.size(), 1.)
                         .cwiseQuotient(Eigen::Map<dVec>(sigmas.data(), sigmas.size()));
        dVec shift = dVec::Constant(sigmas.size(), 0.);
        return {scaling, shift, lbounds.data(), ubounds.data()};
    }

    void initialize()
    {
        std::vector<double> x0 = param_x0();
        assert(x0.size() == param_lbounds().size());
        genopheno_ = make_genopheno();
        cmaparams_ = cma::CMAParameters<GenoPheno>{
          x0, config_.at("opt.sigma").as<double>(), config_.at("opt.lambda").as<int>()};
        cmaparams_.set_gp(genopheno_);
        cmaparams_.set_stopping_criteria(cma::CMAStopCritType::TOLHISTFUN, false);
        cmaparams_.set_stopping_criteria(cma::CMAStopCritType::STAGNATION, false);
        cmaparams_.set_stopping_criteria(cma::CMAStopCritType::NOEFFECTAXIS, false);
        cmaparams_.set_stopping_criteria(cma::CMAStopCritType::NOEFFECTCOOR, false);
        // The following constructor is broken, it sets sigma to the minimum of the sigmas,
        // while it optimizes on interval [0, 10].
        // cmaparams_ = cma::CMAParameters<GenoPheno>
        //   {x0, sigmas, config_.at("opt.lambda").as<int>(), lbounds, ubounds,
        //    prng_->operator()()};
        cmaparams_.set_str_algo(config_.at("opt.algorithm").as<std::string>());
        cmaparams_.set_restarts(config_.at("opt.restarts").as<int>());
        if (config_.count("opt.cmaes-fplot")) {
            cmaparams_.set_fplot(config_.at("opt.cmaes-fplot").as<std::string>());
        }
        cmaparams_.set_mt_feval(!no_multithreading_);
        cmaparams_.set_elitism(config_.at("opt.elitism").as<int>());
        cmaparams_.set_max_fevals(config_.at("opt.max-fevals").as<int>());
        if (config_.at("opt.uncertainty").as<bool>()) cmaparams_.set_uh(true);
    }

    cma::CMASolutions optimize()
    {
        initialize();
        auto fit_func = build_fit_func_();
        auto progress_func = build_progress_func_();
        cma::CMASolutions cmasols = cma::cmaes<GenoPheno>(fit_func, cmaparams_, progress_func);
        progress_func(cmaparams_, cmasols);
        std::cout << "optimization took " << cmasols.elapsed_time() / 1000.0 << " seconds\n";
        return cmasols;
    }

    /// The best evaluation of a single network.
    EvaluationResult& best_evaluation()
    {
        return best_evaluation_;
    }

    /// The evaluation function.
    virtual EvaluationResult
    evaluate(const std::vector<double>& params, std::mt19937& prng) const = 0;

    /// The lower bounds for the optimized parameters.
    virtual std::vector<double> param_lbounds() const = 0;
    /// The upper bounds for the optimized parameters.
    virtual std::vector<double> param_ubounds() const = 0;

    // Find a good set of initial parameters using the bounds.
    virtual std::vector<double> param_x0() const = 0;

    /// Sigmas for individual parameters.
    virtual std::vector<double> param_sigmas() const
    {
        return std::vector<double>(param_lbounds().size(), 1.0);
    }

    std::vector<double> pheno_candidate(const dVec& candidate) const
    {
        // scale the candidate to the proper space using genopheno.
        auto eigen_x = genopheno_.pheno(candidate);
        // convert to std::vector
        std::size_t len = eigen_x.cols() * eigen_x.rows();
        return std::vector<double>(eigen_x.data(), eigen_x.data() + len);
    }

    std::vector<double> pheno_candidate(const cma::Candidate& candidate) const
    {
        return pheno_candidate(candidate.get_x_dvec());
    }

    /// Get the f_value for the given parameter vector.
    double f_value(const std::vector<double>& params, std::mt19937& prng)
    {
        auto evaluate_and_update_best = [&](double& fv) {
            EvaluationResult er = this->evaluate(params, prng);
            fv = er.f_value;
            std::scoped_lock sl{best_evaluation_mutex_};
            if (er.f_value < best_evaluation_.f_value) best_evaluation_ = std::move(er);
        };

        std::vector<double> results(n_evals_);
        // Note no mulithreading because we want to have the same random
        // sequence in all the f_value() calls.
        std::for_each(
          std::execution::seq, results.begin(), results.end(), evaluate_and_update_best);

        if (f_value_agg_ == "mean") {
            return stats{results}.mean();
        } else if (f_value_agg_ == "median") {
            return stats{results}.median();
        } else if (f_value_agg_ == "max") {
            return stats{results}.max();
        } else {
            throw std::invalid_argument{"Invalid f-value aggregate `" + f_value_agg_ + "`."};
        }
    }

    double f_value(const dVec& candidate, std::mt19937& prng)
    {
        return f_value(pheno_candidate(candidate), prng);
    }

    virtual std::ostream& print_params(std::ostream& out, const std::vector<double>& params) const
    {
        for (std::size_t i = 0; i < params.size(); ++i) {
            out << "\tparam " << i << ": "
                << std::setprecision(std::numeric_limits<double>::max_digits10) << params.at(i);
            if (i + 1 < params.size()) out << '\n';
        }
        return out;
    }

    std::ostream& print_candidate(std::ostream& out, const cma::Candidate& candidate) const
    {
        auto out_precision = out.precision();
        out.precision(std::numeric_limits<double>::max_digits10);
        // print f-value.
        out << "f-value: " << candidate.get_fvalue() << '\n';
        // scale the candidate to the proper space using genopheno.
        std::vector<double> params = pheno_candidate(candidate);
        print_params(out, params);
        // restore the oritinal precision
        out.precision(out_precision);
        return out;
    }

    std::ostream& print_candidate(std::ostream& out, const dVec& vec, std::mt19937& prng)
    {
        return print_candidate(out, cma::Candidate{f_value(vec, prng), vec});
    }

    virtual ~optimizer() = default;
};

struct net_evaluation_result_t {
    double f_value;
    std::vector<double> params;
    std::unique_ptr<net_base> net;
};

class net_optimizer : public optimizer<net_evaluation_result_t> {
protected:
    std::unique_ptr<benchmark_set_base> bench_;
    int af_device_;

    virtual std::string arg_prefix_() const = 0;

    virtual double exp_transform(double v) const
    {
        return std::exp(-50.0 * v);
    }

    virtual double inv_exp_transform(double v) const
    {
        return -std::log(v) / 50.;
    }

    virtual double pow_transform(double v) const
    {
        return 2 * v * std::abs(v);
    }

    virtual double inv_pow_transform(double v) const
    {
        return std::copysign(std::sqrt(std::abs(v) / 2.), v);
    }

public:
    virtual po::variables_map to_variables_map(const std::vector<double>& params) const
    {
        // TODO this is ugly, make each optimizer list those that it optimizes instead and
        // put all the defaults to net_optimizer (or even better - take those from params)
        assert(params.size() == 8);
        std::string p = arg_prefix_();
        po::variables_map cfg = config_;
        auto val = [](double v) { return po::variable_value{v, false}; };  // syntactic sugar
        auto expval = [&](double v) { return val(exp_transform(v)); };
        auto powval = [&](double v) { return val(pow_transform(v)); };
        cfg.insert_or_assign(p + "sigma-res", expval(params.at(0)));
        cfg.insert_or_assign(p + "mu-res", powval(params.at(1)));
        cfg.insert_or_assign(p + "in-weight", powval(params.at(2)));
        cfg.insert_or_assign(p + "fb-weight", powval(params.at(3)));
        cfg.insert_or_assign(p + "sparsity", val(std::clamp(params.at(4), 0.0, 1.0)));
        cfg.insert_or_assign(p + "leakage", val(std::clamp(params.at(5), 0.0, 1.0)));
        cfg.insert_or_assign(p + "noise", expval(params.at(6)));
        cfg.insert_or_assign(p + "mu-b", powval(params.at(7)));
        return cfg;
    }

    std::ostream& print_params(std::ostream& out, const std::vector<double>& params) const override
    {
        return out << to_variables_map(params);
    }

    net_optimizer() = default;

    net_optimizer(
      po::variables_map config, std::unique_ptr<benchmark_set_base> bench, std::mt19937& prng)
      : optimizer{std::move(config), prng}
      , bench_{std::move(bench)}
      , af_device_{config_.at("gen.af-device").as<int>()}
    {
    }

    virtual std::unique_ptr<net_base>
    make_net(const std::vector<double>& params, std::mt19937& prng) const = 0;

    virtual double evaluate(net_base& net, std::mt19937& prng) const
    {
        return bench_->evaluate(net, prng);
    }

    net_evaluation_result_t
    evaluate(const std::vector<double>& params, std::mt19937& prng) const override
    {
        auto net = this->make_net(params, prng);
        double f_value = evaluate(*net, prng);
        return {.f_value = f_value, .params = params, .net = std::move(net)};
    }
};

class lcnn_optimizer : public net_optimizer {
protected:
    std::string arg_prefix_() const override
    {
        return "lcnn.";
    }

    std::vector<double> param_x0_;
    double neuron_ins_;
    double init_sigma_res_;

public:
    lcnn_optimizer(
      po::variables_map config, std::unique_ptr<benchmark_set_base> bench, std::mt19937& prng)
      : net_optimizer{std::move(config), std::move(bench), prng}
    {
        // Deduce initial sigma-res from the number of connections to each neuron.
        double unit_sigma = inv_exp_transform(1.0);
        param_x0_ = {unit_sigma, 0.0, 0.1, 0.0, 0.1, 0.9, 0.2, 0.0};
        std::unique_ptr<net_base> sample_net = make_net(param_x0_, prng);
        neuron_ins_ = sample_net->neuron_ins();
        // Set initial sigma-res.
        param_x0_.at(0) = inv_exp_transform(1. / std::sqrt(2 * neuron_ins_));
        // Sparse nets should be biased towards positive mu_res, e.g. 0.3, negative mu-res provide
        // slightly worse results than positive mu-res.
        if (neuron_ins_ < 5.) param_x0_.at(1) = inv_pow_transform(0.3);
    }

    std::unique_ptr<net_base>
    make_net(const std::vector<double>& params, std::mt19937& prng) const override
    {
        af::setDevice(af_device_);
        po::variables_map cfg = to_variables_map(params);
        return std::make_unique<lcnn<af::dtype::f64>>(
          random_lcnn(bench_->n_ins(), bench_->n_outs(), cfg, prng));
    }

    std::vector<double> param_x0() const override
    {
        return param_x0_;
    }

    std::vector<double> param_sigmas() const override
    {
        return {0.01, 0.05, 0.05, 0.01, 0.05, 0.05, 0.05, 0.05};
    }

    std::vector<double> param_lbounds() const override
    {
        return {-1.1, -1.1, -1.1, -1.1, -0.1, -0.1, -0.1, -1.1};
    }

    std::vector<double> param_ubounds() const override
    {
        return {1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1};
    }
};

/// LCNN optimizer without noise.
class lcnn_noiseless_optimizer : public lcnn_optimizer {
public:
    using lcnn_optimizer::lcnn_optimizer;

    po::variables_map to_variables_map(const std::vector<double>& params) const override
    {
        std::vector<double> params_ = params;
        params_.insert(params_.begin() + 6, std::numeric_limits<double>::infinity());
        return lcnn_optimizer::to_variables_map(params_);
    }

    std::vector<double> param_x0() const override
    {
        std::vector<double> x0 = lcnn_optimizer::param_x0();
        x0.erase(x0.begin() + 6);
        return x0;
    }

    std::vector<double> param_sigmas() const override
    {
        std::vector<double> sigmas = lcnn_optimizer::param_sigmas();
        sigmas.erase(sigmas.begin() + 6);
        return sigmas;
    }

    std::vector<double> param_lbounds() const override
    {
        std::vector<double> lbounds = lcnn_optimizer::param_lbounds();
        lbounds.erase(lbounds.begin() + 6);
        return lbounds;
    }

    std::vector<double> param_ubounds() const override
    {
        std::vector<double> ubounds = lcnn_optimizer::param_ubounds();
        ubounds.erase(ubounds.begin() + 6);
        return ubounds;
    }
};

/// LCNN optimizer without feedback weights.
class lcnn_nofb_optimizer : public lcnn_optimizer {
public:
    using lcnn_optimizer::lcnn_optimizer;

    po::variables_map to_variables_map(const std::vector<double>& params) const override
    {
        std::vector<double> params_ = params;
        params_.insert(params_.begin() + 3, 0.0);
        return lcnn_optimizer::to_variables_map(params_);
    }

    std::vector<double> param_x0() const override
    {
        std::vector<double> x0 = lcnn_optimizer::param_x0();
        x0.erase(x0.begin() + 3);
        return x0;
    }

    std::vector<double> param_sigmas() const override
    {
        std::vector<double> sigmas = lcnn_optimizer::param_sigmas();
        sigmas.erase(sigmas.begin() + 3);
        return sigmas;
    }

    std::vector<double> param_lbounds() const override
    {
        std::vector<double> lbounds = lcnn_optimizer::param_lbounds();
        lbounds.erase(lbounds.begin() + 3);
        return lbounds;
    }

    std::vector<double> param_ubounds() const override
    {
        std::vector<double> ubounds = lcnn_optimizer::param_ubounds();
        ubounds.erase(ubounds.begin() + 3);
        return ubounds;
    }
};

class esn_optimizer : public net_optimizer {
protected:
    std::string arg_prefix_() const override
    {
        return "esn.";
    }

public:
    using net_optimizer::net_optimizer;

    std::unique_ptr<net_base>
    make_net(const std::vector<double>& params, std::mt19937& prng) const override
    {
        af::setDevice(af_device_);
        po::variables_map cfg = to_variables_map(params);
        return std::make_unique<simple_esn<af::dtype::f64>>(
          random_esn(bench_->n_ins(), bench_->n_outs(), cfg, prng));
    }

    std::vector<double> param_x0() const override
    {
        return {0.12, 0.0, 0.1, 0.0, 0.5, 0.9};
    }

    std::vector<double> param_sigmas() const override
    {
        return std::vector<double>(6, 0.01);
    }

    std::vector<double> param_lbounds() const override
    {
        return std::vector<double>(6, -1.1);
    }

    std::vector<double> param_ubounds() const override
    {
        return std::vector<double>(6, 1.1);
    }
};

po::options_description optimizer_arg_description()
{
    po::options_description optimizer_arg_desc{"Optimizer options"};
    optimizer_arg_desc.add_options()                                                        //
      ("opt.cmaes-fplot", po::value<std::string>()->default_value("fplot.dat"),             //
       "Output file of the CMA-ES optimization plot.")                                      //
      ("opt.max-fevals", po::value<int>()->default_value(5000),                             //
       "Set the maximum number of evaluations of the objective function "                   //
       "aggregated evals are considered as one).")                                          //
      ("opt.lambda", po::value<int>()->default_value(25),                                   //
       "The number of offspring sampled in each step. Use -1 for automatic deduction.")     //
      ("opt.sigma", po::value<double>()->default_value(2),                                  //
       "The initial sigma value, i.e., the level of exploration. Beware that internally, "  //
       "the optimized range is [0, 10].")                                                   //
      ("opt.uncertainty", po::value<bool>()->default_value(false),                          //
       "Set up uncertainty handling.")                                                      //
      ("opt.n-evals", po::value<int>()->default_value(1),                                   //
       "Run the evaluation function multiple times and aggregate those by f-value-agg.")    //
      ("opt.f-value-agg", po::value<std::string>()->default_value("median"),                //
       "The aggregate function for the multiple evaluations (see n-evals).")                //
      ("opt.algorithm", po::value<std::string>()->default_value("acmaes"),                  //
       "The algorithm to be used. One of {cmaes,ipop,bipop,acmaes,aipop,abipop,sepcmaes,"   //
       "sepipop,sepbipop,sepacmaes,sepaipop,sepabipop,vdcma,vdipopcma,vdbipopcma}")         //
      ("opt.restarts", po::value<int>()->default_value(9),                                  //
       "The maximum number of restarts of the IPOP or BIPOP algorithm.")                    //
      ("opt.elitism", po::value<int>()->default_value(0),                                   //
       "Elitism mode. 0 -> disabled, 1 -> reinject the best, 2 -> reinject x0 "             //
       "till improvement, 3 -> restart if the best encountered solution "                   //
       "is not the final solution.")                                                        //
      ("opt.no-multithreading", po::bool_switch(),                                          //
       "Do not evaluate the individuals in the population in parallel.");                   //
    return optimizer_arg_desc;
}

std::unique_ptr<net_optimizer> make_optimizer(
  std::unique_ptr<benchmark_set_base> bench, const po::variables_map& args, std::mt19937& prng)
{
    if (args.at("gen.net-type").as<std::string>() == "lcnn") {
        if (args.at("gen.optimizer-type").as<std::string>() == "lcnn") {
            return std::make_unique<lcnn_optimizer>(args, std::move(bench), prng);
        }
        if (args.at("gen.optimizer-type").as<std::string>() == "lcnn-nofb") {
            return std::make_unique<lcnn_nofb_optimizer>(args, std::move(bench), prng);
        }
        if (args.at("gen.optimizer-type").as<std::string>() == "lcnn-noiseless") {
            return std::make_unique<lcnn_noiseless_optimizer>(args, std::move(bench), prng);
        }
        throw std::invalid_argument{"Unknown lcnn optimizer type."};
    }
    if (args.at("gen.net-type").as<std::string>() == "simple-esn") {
        return std::make_unique<esn_optimizer>(args, std::move(bench), prng);
    }
    throw std::runtime_error{
      "Unknown net type \"" + args.at("gen.net-type").as<std::string>() + "\".\n"};
}

}  // namespace esn
