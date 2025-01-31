#pragma once

// Echo state network optimization header. //

#include "argument_utils.hpp"
#include "benchmark_results.hpp"
#include "benchmarks.hpp"
#include "lcnn.hpp"
#include "lcnn_ensemble.hpp"
#include "net.hpp"

#include <boost/program_options.hpp>
#include <execution>
#include <filesystem>
#include <functional>
#include <iostream>
#include <libcmaes/cmaes.h>
#include <limits>
#include <map>
#include <mutex>
#include <range/v3/all.hpp>

namespace esn {

namespace po = boost::program_options;
namespace rg = ranges;
namespace rga = ranges::actions;
namespace rgv = ranges::views;

inline const std::vector<std::string> DEFAULT_EXCLUDED_PARAMS = {
  "lcnn.mu-in-weight",
  "lcnn.mu-fb-weight",
  "lcnn.sigma-b",
  "lcnn.noise",
  "lcnn.sparsity",
  "lcnn.in-fb-sparsity",
  "lcnn.leakage",
  "lcnn.l2",
  "lcnn.enet-alpha",
  "lcnn.n-state-predictors",
  "lcnn.train-valid-ratio",
  "lcnn.act-steepness",
  "lcnn.input-to-n",
  "lcnn.memory-prob",
  "lcnn.sigma-memory",
  "lcnn.mu-memory",
  "lcnn.lms-mu",
  "lcnn.adapt.learning-rate",
  "lcnn.adapt.weight-leakage",
  "lcnn.adapt.abs-target-activation",
  "esn.noise",
  "esn.sparsity"};
inline const std::string DEFAULT_EXCLUDED_PARAMS_STR =
  rgv::join(DEFAULT_EXCLUDED_PARAMS, ',') | rg::to<std::string>();

/// Generic function optimizer.
template <typename EvaluationResult>
class optimizer {
protected:
    po::variables_map config_;
    using GenoPheno = cma::GenoPheno<cma::pwqBoundStrategy, cma::linScalingStrategy>;
    GenoPheno genopheno_;
    cma::CMAParameters<GenoPheno> cmaparams_;
    int n_evals_;
    optimization_status_t opt_status_;
    std::string f_value_agg_;
    bool multithreading_;
    bool reseed_every_epoch_;
    EvaluationResult best_evaluation_;
    std::mutex best_evaluation_mutex_;
    prng_t prng_;
    fs::path output_dir_;

    virtual void
    progress(const cma::CMAParameters<GenoPheno>& cmaparams, const cma::CMASolutions& cmasols)
    {
        return;
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
              std::unique_lock ul{best_evaluation_mutex_};
              if (best_evaluation_.net) best_evaluation_.net->save(output_dir_ / "best-model");
              opt_status_ = {.progress = (double)cmasols.nevals() / cmaparams.get_max_fevals()};
              progress(cmaparams, cmasols);
              std::cout << std::endl;
              if (reseed_every_epoch_) reseed(prng_, 137);
              return 0;
          };
    }

    /// Build the FitFunc for libcmaes.
    cma::FitFunc build_fit_func_()
    {
        return [this](const double* p, int n) -> double {
            std::vector<double> params(p, p + n);
            prng_t prng = prng_;  // copy the generator for each f_value() call.
            return f_value(params, prng);
        };
    }

public:
    /// Dummy constructor.
    optimizer() = default;
    /// Construct from configuration.
    optimizer(po::variables_map config, prng_t prng, fs::path output_dir)
      : config_{std::move(config)}
      , n_evals_{config_.at("opt.n-evals").as<int>()}
      , f_value_agg_{config_.at("opt.f-value-agg").as<std::string>()}
      , multithreading_{config_.at("opt.multithreading").as<bool>()}
      , reseed_every_epoch_{config_.at("opt.reseed-every-epoch").as<bool>()}
      , prng_{std::move(prng)}
      , output_dir_{std::move(output_dir)}
    {
        fs::create_directories(output_dir_);
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
        reseed(prng_, 137);
        clear_best_evaluation();
        std::vector<double> x0 = param_x0();
        assert(x0.size() == param_lbounds().size());
        opt_status_ = {.progress = 0.};
        genopheno_ = make_genopheno();
        cmaparams_ = cma::CMAParameters<GenoPheno>{
          x0, config_.at("opt.sigma").as<double>(), config_.at("opt.lambda").as<int>()};
        cmaparams_.set_gp(genopheno_);
        cmaparams_.set_stopping_criteria(cma::CMAStopCritType::TOLHISTFUN, false);
        cmaparams_.set_stopping_criteria(cma::CMAStopCritType::TOLX, false);
        cmaparams_.set_stopping_criteria(cma::CMAStopCritType::TOLUPSIGMA, false);
        cmaparams_.set_stopping_criteria(cma::CMAStopCritType::STAGNATION, false);
        cmaparams_.set_stopping_criteria(cma::CMAStopCritType::NOEFFECTAXIS, false);
        cmaparams_.set_stopping_criteria(cma::CMAStopCritType::NOEFFECTCOOR, false);
        cmaparams_.set_seed(prng_());
        // The following constructor is broken, it sets sigma to the minimum of the sigmas,
        // while it optimizes on interval [0, 10].
        // cmaparams_ = cma::CMAParameters<GenoPheno>
        //   {x0, sigmas, config_.at("opt.lambda").as<int>(), lbounds, ubounds, prng_()};
        cmaparams_.set_str_algo(config_.at("opt.algorithm").as<std::string>());
        cmaparams_.set_restarts(config_.at("opt.restarts").as<int>());
        cmaparams_.set_fplot(output_dir_ / "fplot.dat");
        cmaparams_.set_mt_feval(multithreading_);
        cmaparams_.set_elitism(config_.at("opt.elitism").as<int>());
        cmaparams_.set_max_fevals(config_.at("opt.max-fevals").as<int>());
        if (config_.at("opt.uncertainty").as<bool>()) cmaparams_.set_uh(true);
        if (config_.at("opt.tpa").as<bool>()) cmaparams_.set_tpa(2);
        if (config_.at("opt.noisy").as<bool>()) cmaparams_.set_noisy();
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

    /// Remove the currently best network from memory and from file storage.
    void clear_best_evaluation()
    {
        fs::remove_all(output_dir_ / "best-model");
        best_evaluation_ = {
          .f_value = std::numeric_limits<double>::infinity(), .params = {}, .net = {}};
    }

    /// The best evaluation of a single network.
    EvaluationResult& best_evaluation()
    {
        return best_evaluation_;
    }

    /// The evaluation function.
    virtual EvaluationResult evaluate(
      const std::vector<double>& params, prng_t& prng, optimization_status_t status) const = 0;

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
    double f_value(const std::vector<double>& params, prng_t& prng)
    {
        auto evaluate_and_update_best = [&](double& fv) {
            EvaluationResult er = this->evaluate(params, prng, opt_status_);
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

    double f_value(const dVec& candidate, prng_t& prng)
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
        // restore the original precision
        out.precision(out_precision);
        return out;
    }

    std::ostream& print_candidate(std::ostream& out, const dVec& vec, prng_t& prng)
    {
        return print_candidate(out, cma::Candidate{f_value(vec, prng), vec});
    }

    virtual ~optimizer() = default;
};

struct net_evaluation_result_t {
    double f_value;
    std::map<std::string, double> params;
    std::unique_ptr<net_base> net;
};

/// Interface for the optimizer providing named access to optimized parameters.
template <typename EvaluationResult>
class named_optimizer : public optimizer<EvaluationResult> {
private:
    EvaluationResult evaluate(
      const std::vector<double>& params, prng_t& prng, optimization_status_t status) const override
    {
        return named_evaluate(name_and_filter_params(params), prng, status);
    }

    std::vector<double> param_x0() const override
    {
        return unname_and_filter_params(named_param_x0());
    }

    std::vector<double> param_sigmas() const override
    {
        return unname_and_filter_params(named_param_sigmas());
    }

    std::vector<double> param_lbounds() const override
    {
        return unname_and_filter_params(named_param_lbounds());
    }

    std::vector<double> param_ubounds() const override
    {
        return unname_and_filter_params(named_param_ubounds());
    }

protected:
    std::set<std::string> excluded_params_;

public:
    named_optimizer() = default;

    named_optimizer(po::variables_map config, prng_t prng, fs::path output_dir)
      : optimizer<EvaluationResult>(std::move(config), std::move(prng), std::move(output_dir))
    {
        if (this->config_.contains("opt.exclude-params")) {
            std::vector<std::string>& excluded_params_arg =
              this->config_.at("opt.exclude-params").template as<std::vector<std::string>>();
            excluded_params_ = excluded_params_arg | rg::to<std::set<std::string>>();
            // Replace `default` keyword with default arguments.
            if (excluded_params_.contains("default")) {
                excluded_params_.erase("default");
                excluded_params_.insert(
                  DEFAULT_EXCLUDED_PARAMS.begin(), DEFAULT_EXCLUDED_PARAMS.end());
            }
            if (this->config_.contains("opt.include-params")) {
                for (const std::string& param :
                     this->config_.at("opt.include-params").template as<std::vector<std::string>>())
                    excluded_params_.erase(param);
            }
            // Update the original argument to have better logs.
            excluded_params_arg = excluded_params_ | rg::to<std::vector<std::string>>();
        }
    }

    std::map<std::string, double> name_and_filter_params(const std::vector<double>& params) const
    {
        std::set<std::string> param_names = available_params();
        for (const std::string& ep : excluded_params_) {
            std::set<std::string> new_param_names;
            for (const std::string& p : param_names)
                if (!p.starts_with(ep)) new_param_names.insert(p);
            param_names = std::move(new_param_names);
        }
        for (const std::set<std::string>& pg : param_groups()) {
            if (pg.empty()) continue;
            std::set<std::string> new_param_names = param_names;
            // only keep the group representant
            for (const std::string& param : param_names)
                if (pg.contains(param) && param != *pg.begin()) new_param_names.erase(param);
            param_names = std::move(new_param_names);
        }
        assert(param_names.size() == params.size());
        std::map<std::string, double> param_map =
          rgv::zip(param_names, params) | rg::to<std::map>();
        for (const std::set<std::string>& pg : param_groups()) {
            if (pg.empty()) continue;
            for (const auto& [param, value] : rgv::zip(param_names, params)) {
                // copy the value from the group representant to all
                // the other group members
                if (param == *pg.begin()) {
                    for (const std::string& dependent_name : pg) {
                        param_map.emplace(dependent_name, value);
                    }
                }
            }
        }
        return param_map;
    }

    std::vector<double> unname_and_filter_params(std::map<std::string, double> params) const
    {
        assert(params.size() == available_params().size());
        for (const std::string& ep : excluded_params_) {
            std::map<std::string, double> new_params;
            for (const auto& [param, value] : params)
                if (!param.starts_with(ep)) new_params.insert({param, value});
            params = std::move(new_params);
        }
        for (const std::set<std::string>& pg : param_groups()) {
            if (pg.empty()) continue;
            std::map<std::string, double> new_params = params;
            // only keep the group representant
            for (const auto& [param, value] : params)
                if (pg.contains(param) && param != *pg.begin()) new_params.erase(param);
            params = std::move(new_params);
        }
        return rgv::values(params) | rg::to_vector;
    }

    virtual std::set<std::string> available_params() const = 0;
    virtual EvaluationResult named_evaluate(
      const std::map<std::string, double>& params,
      prng_t& prng,
      optimization_status_t status) const = 0;
    virtual std::map<std::string, double> named_param_x0() const = 0;
    virtual std::map<std::string, double> named_param_sigmas() const = 0;
    virtual std::map<std::string, double> named_param_lbounds() const = 0;
    virtual std::map<std::string, double> named_param_ubounds() const = 0;
    virtual std::vector<std::set<std::string>> param_groups() const = 0;
};

class net_optimizer : public named_optimizer<net_evaluation_result_t> {
private:
    net_evaluation_result_t named_evaluate(
      const std::map<std::string, double>& params,
      prng_t& prng,
      optimization_status_t status) const override
    {
        auto net = this->make_net(params, prng);
        double f_value = evaluate_net(*net, prng, status);
        return {.f_value = f_value, .params = params, .net = std::move(net)};
    }

protected:
    benchmark_factory_t bench_factory_;
    std::unique_ptr<benchmark_set_base> bench_;
    int af_device_;
    nlohmann::json param_stages_;
    double weight_cutoff_;
    bool in_fb_group_;
    std::string error_measure_;

    virtual std::string arg_prefix_() const = 0;

    static double exp_transform(double v)
    {
        return std::exp(-50.0 * v);
    }

    static double inv_exp_transform(double v)
    {
        return -std::log(v) / 50.;
    }

    static double pow_transform(double v)
    {
        return 2 * v * std::abs(v);
    }

    static double inv_pow_transform(double v)
    {
        return std::copysign(std::sqrt(std::abs(v) / 2.), v);
    }

public:
    const std::string& error_measure() const
    {
        return error_measure_;
    }

    po::variables_map to_variables_map(const std::vector<double>& params) const
    {
        return to_variables_map(name_and_filter_params(params));
    }

    std::map<std::string, double>
    from_variables_map(const std::set<std::string>& keys, const po::variables_map& vm) const
    {
        std::map<std::string, double> params;
        for (const std::string& key : keys) {
            const std::string& p = arg_prefix_();
            po::variables_map cfg = config_;
            std::string mu_in_weight_prefix = p + "mu-in-weight-";
            std::string sigma_in_weight_prefix = p + "sigma-in-weight-";
            std::string mu_fb_weight_prefix = p + "mu-fb-weight-";
            std::string sigma_fb_weight_prefix = p + "sigma-fb-weight-";
            if (key == p + "sigma-res") {
                params.emplace(key, inv_exp_transform(vm.at(key).as<double>()));
            } else if (key == p + "mu-res") {
                params.emplace(key, inv_pow_transform(vm.at(key).as<double>()));
            } else if (key.starts_with(mu_in_weight_prefix)) {
                long idx = std::stol(key.substr(mu_in_weight_prefix.length()));
                std::vector<double> mu_in_weights =
                  vm.at(p + "mu-in-weight").as<std::vector<double>>();
                params.emplace(key, inv_pow_transform(mu_in_weights.at(idx)));
            } else if (key.starts_with(sigma_in_weight_prefix)) {
                long idx = std::stol(key.substr(sigma_in_weight_prefix.length()));
                std::vector<double> sigma_in_weights =
                  vm.at(p + "sigma-in-weight").as<std::vector<double>>();
                params.emplace(key, inv_exp_transform(std::max(sigma_in_weights.at(idx), 1e-40)));
            } else if (key.starts_with(mu_fb_weight_prefix)) {
                long idx = std::stol(key.substr(mu_fb_weight_prefix.length()));
                std::vector<double> mu_fb_weights =
                  vm.at(p + "mu-fb-weight").as<std::vector<double>>();
                params.emplace(key, inv_pow_transform(mu_fb_weights.at(idx)));
            } else if (key.starts_with(sigma_fb_weight_prefix)) {
                long idx = std::stol(key.substr(sigma_fb_weight_prefix.length()));
                std::vector<double> sigma_fb_weights =
                  vm.at(p + "sigma-fb-weight").as<std::vector<double>>();
                params.emplace(key, inv_exp_transform(std::max(sigma_fb_weights.at(idx), 1e-40)));
            } else if (key == p + "sparsity") {
                params.emplace(key, vm.at(key).as<double>());
            } else if (key == p + "in-fb-sparsity") {
                params.emplace(key, vm.at(key).as<double>());
            } else if (key == p + "leakage") {
                params.emplace(key, vm.at(key).as<double>());
            } else if (key == p + "noise") {
                params.emplace(key, inv_exp_transform(std::max(vm.at(key).as<double>(), 1e-40)));
            } else if (key == p + "sigma-b") {
                params.emplace(key, inv_exp_transform(std::max(vm.at(key).as<double>(), 1e-40)));
            } else if (key == p + "mu-b") {
                params.emplace(key, inv_pow_transform(vm.at(key).as<double>()));
            } else if (key == "lcnn.l2") {
                params.emplace(key, inv_exp_transform(std::max(vm.at(key).as<double>(), 1e-40)));
            } else if (key == "lcnn.enet-lambda") {
                params.emplace(key, inv_exp_transform(std::max(vm.at(key).as<double>(), 1e-40)));
            } else if (key == "lcnn.enet-alpha") {
                params.emplace(key, vm.at(key).as<double>());
            } else if (key == "lcnn.n-state-predictors") {
                params.emplace(key, vm.at(key).as<double>());
            } else if (key == "lcnn.train-valid-ratio") {
                params.emplace(key, vm.at(key).as<double>());
            } else if (key == "lcnn.input-to-n") {
                params.emplace(key, vm.at(key).as<double>());
            } else if (key == p + "act-steepness") {
                params.emplace(key, inv_pow_transform(vm.at(key).as<double>()));
            } else if (key == "lcnn.memory-prob") {
                params.emplace(key, vm.at(key).as<double>());
            } else if (key == "lcnn.sigma-memory") {
                params.emplace(key, vm.at(key).as<double>());
            } else if (key == "lcnn.mu-memory") {
                params.emplace(key, vm.at(key).as<double>());
            } else if (key == "lcnn.lms-mu") {
                params.emplace(key, inv_exp_transform(vm.at(key).as<double>()));
            } else if (key == "lcnn.adapt.learning-rate") {
                params.emplace(key, inv_exp_transform(vm.at(key).as<double>()));
            } else if (key == "lcnn.adapt.weight-leakage") {
                params.emplace(key, inv_exp_transform(vm.at(key).as<double>()));
            } else if (key == "lcnn.adapt.abs-target-activation") {
                params.emplace(key, inv_exp_transform(vm.at(key).as<double>()));
            }
        }
        return params;
    }

    po::variables_map patch_via_stages(po::variables_map vm) const
    {
        if (param_stages_.empty()) return vm;
        for (const auto& [progress, params] : param_stages_.items()) {
            if (opt_status_.progress >= std::stod(progress)) {
                for (const auto& [key, value] : params.items()) {
                    if (key.starts_with("_")) continue;
                    if (value.is_number_integer())
                        vm.insert_or_assign(key, po::variable_value{value.get<long>(), false});
                    else if (value.is_number_float())
                        vm.insert_or_assign(key, po::variable_value{value.get<double>(), false});
                    else if (value.is_string())
                        vm.insert_or_assign(
                          key, po::variable_value{value.get<std::string>(), false});
                    else if (value.is_boolean())
                        vm.insert_or_assign(key, po::variable_value{value.get<bool>(), false});
                    else
                        throw std::runtime_error{"Unsupported type in param stages file."};
                }
            }
        }
        return vm;
    }

    po::variables_map to_variables_map(std::map<std::string, double> params) const
    {
        // syntactic sugar
        auto val = [](double v) { return po::variable_value{v, false}; };
        auto expval = [&](double v) { return val(exp_transform(v)); };
        auto powval = [&](double v) { return val(pow_transform(v)); };
        auto vector_val = [&params](const std::string& name, auto& proj) {
            std::vector<double> values;
            for (int i = 0; true; ++i) {
                std::string param_name = name + "-" + std::to_string(i);
                if (params.contains(param_name)) {
                    values.push_back(proj(params.at(param_name)));
                    params.erase(param_name);
                } else {
                    break;
                }
            }
            return values;
        };

        const std::string& p = arg_prefix_();
        po::variables_map cfg = config_;

        if (params.contains(p + "sigma-res")) {
            cfg.insert_or_assign(p + "sigma-res", expval(params.at(p + "sigma-res")));
            params.erase(p + "sigma-res");
        }
        if (params.contains(p + "mu-res")) {
            cfg.insert_or_assign(p + "mu-res", powval(params.at(p + "mu-res")));
            params.erase(p + "mu-res");
        }
        std::vector<double> mu_in_weight = vector_val(p + "mu-in-weight", pow_transform);
        if (!mu_in_weight.empty()) {
            cfg.insert_or_assign(p + "mu-in-weight", po::variable_value{mu_in_weight, false});
        }
        std::vector<double> sigma_in_weight = vector_val(p + "sigma-in-weight", exp_transform);
        if (!sigma_in_weight.empty()) {
            rga::transform(sigma_in_weight, [this](double v) -> double {
                if (std::abs(v) < weight_cutoff_) return 0.;
                return v;
            });
            cfg.insert_or_assign(p + "sigma-in-weight", po::variable_value{sigma_in_weight, false});
        }
        std::vector<double> mu_fb_weight = vector_val(p + "mu-fb-weight", pow_transform);
        if (!mu_fb_weight.empty()) {
            cfg.insert_or_assign(p + "mu-fb-weight", po::variable_value(mu_fb_weight, false));
        }
        std::vector<double> sigma_fb_weight = vector_val(p + "sigma-fb-weight", exp_transform);
        if (!sigma_fb_weight.empty()) {
            rga::transform(sigma_fb_weight, [this](double v) -> double {
                if (std::abs(v) < weight_cutoff_) return 0.;
                return v;
            });
            cfg.insert_or_assign(p + "sigma-fb-weight", po::variable_value(sigma_fb_weight, false));
        }
        if (params.contains(p + "sparsity")) {
            cfg.insert_or_assign(
              p + "sparsity", val(std::clamp(params.at(p + "sparsity"), 0.0, 1.0)));
            params.erase(p + "sparsity");
        }
        if (params.contains(p + "in-fb-sparsity")) {
            cfg.insert_or_assign(
              p + "in-fb-sparsity", val(std::clamp(params.at(p + "in-fb-sparsity"), 0.0, 1.0)));
            params.erase(p + "in-fb-sparsity");
        }
        if (params.contains(p + "leakage")) {
            cfg.insert_or_assign(p + "leakage", val(params.at(p + "leakage")));
            params.erase(p + "leakage");
        }
        if (params.contains(p + "noise")) {
            cfg.insert_or_assign(p + "noise", expval(params.at(p + "noise")));
            params.erase(p + "noise");
        }
        if (params.contains(p + "sigma-b")) {
            cfg.insert_or_assign(p + "sigma-b", expval(params.at(p + "sigma-b")));
            params.erase(p + "sigma-b");
        }
        if (params.contains(p + "mu-b")) {
            cfg.insert_or_assign(p + "mu-b", powval(params.at(p + "mu-b")));
            params.erase(p + "mu-b");
        }
        if (params.contains("lcnn.l2")) {
            cfg.insert_or_assign("lcnn.l2", expval(params.at("lcnn.l2")));
            params.erase("lcnn.l2");
        }
        if (params.contains("lcnn.enet-lambda")) {
            cfg.insert_or_assign("lcnn.enet-lambda", expval(params.at("lcnn.enet-lambda")));
            params.erase("lcnn.enet-lambda");
        }
        if (params.contains("lcnn.enet-alpha")) {
            cfg.insert_or_assign(
              "lcnn.enet-alpha", val(std::clamp(params.at("lcnn.enet-alpha"), 0.0, 1.0)));
            params.erase("lcnn.enet-alpha");
        }
        if (params.contains("lcnn.n-state-predictors")) {
            double n_predictors = std::clamp(params.at("lcnn.n-state-predictors"), 0.0, 1.0);
            cfg.insert_or_assign("lcnn.n-state-predictors", val(n_predictors));
            params.erase("lcnn.n-state-predictors");
        }
        if (params.contains("lcnn.train-valid-ratio")) {
            cfg.insert_or_assign(
              "lcnn.train-valid-ratio",
              val(std::clamp(params.at("lcnn.train-valid-ratio"), 0.01, 0.99)));
            params.erase("lcnn.train-valid-ratio");
        }
        if (params.contains("lcnn.input-to-n")) {
            double input_to_n = std::clamp(params.at("lcnn.input-to-n"), 0.0, 1.0);
            cfg.insert_or_assign("lcnn.input-to-n", val(input_to_n));
            params.erase("lcnn.input-to-n");
        }
        if (params.contains(p + "act-steepness")) {
            cfg.insert_or_assign(p + "act-steepness", powval(params.at(p + "act-steepness")));
            params.erase(p + "act-steepness");
        }
        if (params.contains("lcnn.memory-prob")) {
            cfg.insert_or_assign(
              "lcnn.memory-prob", val(std::clamp(params.at("lcnn.memory-prob"), 0.0, 1.0)));
            params.erase("lcnn.memory-prob");
        }
        if (params.contains("lcnn.sigma-memory")) {
            cfg.insert_or_assign("lcnn.sigma-memory", val(params.at("lcnn.sigma-memory")));
            params.erase("lcnn.sigma-memory");
        }
        if (params.contains("lcnn.mu-memory")) {
            cfg.insert_or_assign("lcnn.mu-memory", val(params.at("lcnn.mu-memory")));
            params.erase("lcnn.mu-memory");
        }
        if (params.contains("lcnn.lms-mu")) {
            cfg.insert_or_assign("lcnn.lms-mu", expval(params.at("lcnn.lms-mu")));
            params.erase("lcnn.lms-mu");
        }
        if (params.contains("lcnn.adapt.learning-rate")) {
            cfg.insert_or_assign(
              "lcnn.adapt.learning-rate", expval(params.at("lcnn.adapt.learning-rate")));
            params.erase("lcnn.adapt.learning-rate");
        }
        if (params.contains("lcnn.adapt.weight-leakage")) {
            cfg.insert_or_assign(
              "lcnn.adapt.weight-leakage", expval(params.at("lcnn.adapt.weight-leakage")));
            params.erase("lcnn.adapt.weight-leakage");
        }
        if (params.contains("lcnn.adapt.abs-target-activation")) {
            cfg.insert_or_assign(
              "lcnn.adapt.abs-target-activation",
              expval(params.at("lcnn.adapt.abs-target-activation")));
            params.erase("lcnn.adapt.abs-target-activation");
        }
        assert(
          params.empty() || (params.size() == 1 && params.contains("none")));  // make sure all
                                                                               // the params have
                                                                               // been consumed
        return patch_via_stages(cfg);
    }

    void progress(
      const cma::CMAParameters<GenoPheno>& cmaparams, const cma::CMASolutions& cmasols) override
    {
        if (best_evaluation_.net) {
            std::cout << "input names: " << rgv::all(best_evaluation_.net->input_names())
                      << std::endl;
        }
        for (const auto& [progress, params] : param_stages_.items()) {
            if (opt_status_.progress >= std::stod(progress)) {
                for (const auto& [key, value] : params.items()) {
                    if (key == "_CLEAR_BEST_MODEL") {
                        if (value.get<bool>()) clear_best_evaluation();
                        param_stages_.at(progress).at(key) = false;
                    } else if (key == "_REINITIALIZE_BENCHMARK") {
                        if (value.get<bool>()) initialize_benchmark();
                        param_stages_.at(progress).at(key) = false;
                    } else if (key.starts_with("_"))
                        throw std::runtime_error{
                          fmt::format("Unknown key `{}` in param stages file.", key)};
                }
            }
        }
    }

    std::ostream& print_params(std::ostream& out, const std::vector<double>& params) const override
    {
        return out << to_variables_map(params);
    }

    std::ostream& print_result(std::ostream& out, const net_evaluation_result_t& result) const
    {
        auto out_precision = out.precision();
        out.precision(std::numeric_limits<double>::max_digits10);
        // print f-value and params.
        out << "f-value: " << result.f_value << '\n';
        out << to_variables_map(result.params) << std::endl;
        // restore the original precision
        out.precision(out_precision);
        return out;
    }

    net_optimizer() = default;

    net_optimizer(
      po::variables_map config, benchmark_factory_t bench_factory, prng_t prng, fs::path output_dir)
      : named_optimizer{std::move(config), std::move(prng), std::move(output_dir)}
      , bench_factory_{std::move(bench_factory)}
      , af_device_{config_.at("gen.af-device").as<int>()}
      , weight_cutoff_{config_.at("opt.weight-cutoff").as<double>()}
      , in_fb_group_{config_.at("opt.in-fb-group").as<bool>()}
      , error_measure_{config_.at("opt.error-measure").as<std::string>()}
    {
        if (config_.contains("opt.param-stages-json")) {
            std::ifstream param_stages_fin{config_.at("opt.param-stages-json").as<std::string>()};
            param_stages_ = nlohmann::json::parse(param_stages_fin, nullptr, true);
        }
        initialize_benchmark();
    }

    virtual std::unique_ptr<net_base>
    make_net(const std::map<std::string, double>& params, prng_t& prng) const = 0;

    void initialize_benchmark()
    {
        bench_ = bench_factory_(patch_via_stages(config_));
    }

    benchmark_set_base& benchmark()
    {
        if (bench_ == nullptr) throw std::runtime_error{"No benchmark set."};
        return *bench_;
    }

    virtual double evaluate_net(net_base& net, prng_t& prng, optimization_status_t status) const
    {
        benchmark_results results = bench_->evaluate(net, prng, status);
        if (!results.contains(error_measure_))
            throw std::invalid_argument{fmt::format(
              "Optimizer error measure `{}` not contained in the computed error measures.",
              error_measure_)};
        return results.at(error_measure_).mean();
    }
};

class lcnn_optimizer : public net_optimizer {
protected:
    std::string arg_prefix_() const override
    {
        return "lcnn.";
    }

    std::map<std::string, double> param_x0_;
    double neuron_ins_;
    double init_sigma_res_;

public:
    lcnn_optimizer(
      po::variables_map config, benchmark_factory_t bench_factory, prng_t prng, fs::path output_dir)
      : net_optimizer{
          std::move(config), std::move(bench_factory), std::move(prng), std::move(output_dir)}
    {
        if (config_.at("opt.x0-from-params").as<bool>()) {
            param_x0_ = from_variables_map(available_params(), config_);
            return;
        }
        // Deduce initial sigma-res from the number of connections to each neuron.
        double unit_sigma = inv_exp_transform(1.0);
        param_x0_ = {
          {"lcnn.sigma-res", unit_sigma},
          {"lcnn.mu-res", 0.0},
          {"lcnn.sparsity", 0.1},
          {"lcnn.in-fb-sparsity", 0.1},
          {"lcnn.leakage", 0.9},
          {"lcnn.noise", 0.2},
          {"lcnn.sigma-b", 0.2},
          {"lcnn.mu-b", 0.0},
          {"lcnn.n-state-predictors", 0.5},
          {"lcnn.train-valid-ratio", 0.8},
          {"lcnn.l2", 0.2},
          {"lcnn.enet-lambda", inv_exp_transform(1e-8)},
          {"lcnn.enet-alpha", 0.5},
          {"lcnn.input-to-n", 0.5},
          {"lcnn.act-steepness", inv_pow_transform(1.0)},
          {"lcnn.memory-prob", 0.1},
          {"lcnn.sigma-memory", 0.5},
          {"lcnn.mu-memory", 1.0},
          {"lcnn.lms-mu", inv_exp_transform(1e-3)},
          {"lcnn.adapt.learning-rate", 0.1},
          {"lcnn.adapt.weight-leakage", 0.5},
          {"lcnn.adapt.abs-target-activation", inv_exp_transform(1.0)},
        };
        for (int i = 0; i < (int)bench_->input_names().size(); ++i) {
            param_x0_.insert({"lcnn.mu-in-weight-" + std::to_string(i), 0.0});
            param_x0_.insert(
              {"lcnn.sigma-in-weight-" + std::to_string(i), inv_exp_transform(5e-6)});
        }
        for (int i = 0; i < (int)bench_->output_names().size(); ++i) {
            param_x0_.insert({"lcnn.mu-fb-weight-" + std::to_string(i), 0.0});
            param_x0_.insert(
              {"lcnn.sigma-fb-weight-" + std::to_string(i), inv_exp_transform(5e-6)});
        }
        prng_t prng_clone{prng_};
        std::unique_ptr<net_base> sample_net = make_net(param_x0_, prng_clone);
        neuron_ins_ = sample_net->neuron_ins();
        // Set initial sigma-res.
        param_x0_.at("lcnn.sigma-res") = inv_exp_transform(1. / std::sqrt(2. * neuron_ins_));
        // Sparse nets should be biased towards positive mu_res, e.g. 0.3, negative mu-res
        // provide slightly worse results than positive mu-res.
        if (neuron_ins_ < 5.) param_x0_.at("lcnn.mu-res") = inv_pow_transform(0.3);
    }

    std::set<std::string> available_params() const override
    {
        std::set<std::string> params = {
          "lcnn.sigma-res",
          "lcnn.mu-res",
          "lcnn.sparsity",
          "lcnn.in-fb-sparsity",
          "lcnn.leakage",
          "lcnn.noise",
          "lcnn.sigma-b",
          "lcnn.mu-b",
          "lcnn.n-state-predictors",
          "lcnn.train-valid-ratio",
          "lcnn.l2",
          "lcnn.enet-lambda",
          "lcnn.enet-alpha",
          "lcnn.input-to-n",
          "lcnn.act-steepness",
          "lcnn.memory-prob",
          "lcnn.sigma-memory",
          "lcnn.mu-memory",
          "lcnn.lms-mu",
          "lcnn.adapt.learning-rate",
          "lcnn.adapt.weight-leakage",
          "lcnn.adapt.abs-target-activation",
        };
        for (int i = 0; i < (int)bench_->input_names().size(); ++i) {
            params.insert("lcnn.mu-in-weight-" + std::to_string(i));
            params.insert("lcnn.sigma-in-weight-" + std::to_string(i));
        }
        for (int i = 0; i < (int)bench_->output_names().size(); ++i) {
            params.insert("lcnn.mu-fb-weight-" + std::to_string(i));
            params.insert("lcnn.sigma-fb-weight-" + std::to_string(i));
        }
        return params;
    }

    std::unique_ptr<net_base>
    make_net(const std::map<std::string, double>& params, prng_t& prng) const override
    {
        af::setDevice(af_device_);
        po::variables_map cfg = to_variables_map(params);
        return std::make_unique<lcnn<>>(
          random_lcnn(bench_->input_names(), bench_->output_names(), cfg, prng));
    }

    std::map<std::string, double> named_param_x0() const override
    {
        return param_x0_;
    }

    std::map<std::string, double> named_param_sigmas() const override
    {
        std::map<std::string, double> params = {
          {"lcnn.sigma-res", 0.01},
          {"lcnn.mu-res", 0.05},
          {"lcnn.sparsity", 0.05},
          {"lcnn.in-fb-sparsity", 0.05},
          {"lcnn.leakage", 0.2},
          {"lcnn.noise", 0.05},
          {"lcnn.sigma-b", 0.05},
          {"lcnn.mu-b", 0.05},
          {"lcnn.n-state-predictors", 0.1},
          {"lcnn.train-valid-ratio", 0.1},
          {"lcnn.l2", 0.05},
          {"lcnn.enet-lambda", 0.01},
          {"lcnn.enet-alpha", 0.1},
          {"lcnn.input-to-n", 0.1},
          {"lcnn.act-steepness", 0.05},
          {"lcnn.memory-prob", 0.1},
          {"lcnn.sigma-memory", 0.1},
          {"lcnn.mu-memory", 0.05},
          {"lcnn.lms-mu", 0.01},
          {"lcnn.adapt.learning-rate", 0.05},
          {"lcnn.adapt.weight-leakage", 0.05},
          {"lcnn.adapt.abs-target-activation", 0.05},
        };
        for (int i = 0; i < (int)bench_->input_names().size(); ++i) {
            params.insert({"lcnn.mu-in-weight-" + std::to_string(i), 0.05});
            params.insert({"lcnn.sigma-in-weight-" + std::to_string(i), 0.01});
        }
        for (int i = 0; i < (int)bench_->output_names().size(); ++i) {
            params.insert({"lcnn.mu-fb-weight-" + std::to_string(i), 0.01});
            params.insert({"lcnn.sigma-fb-weight-" + std::to_string(i), 0.01});
        }
        return params;
    }

    std::map<std::string, double> named_param_lbounds() const override
    {
        std::map<std::string, double> params = {
          {"lcnn.sigma-res", -0.1},
          {"lcnn.mu-res", -1.1},
          {"lcnn.sparsity", -0.1},
          {"lcnn.in-fb-sparsity", -0.1},
          {"lcnn.leakage", -2.1},
          {"lcnn.noise", -0.1},
          {"lcnn.sigma-b", -0.1},
          {"lcnn.mu-b", -1.1},
          {"lcnn.n-state-predictors", -0.1},
          {"lcnn.train-valid-ratio", -0.1},
          {"lcnn.l2", -0.1},
          {"lcnn.enet-lambda", -0.1},
          {"lcnn.enet-alpha", -0.1},
          {"lcnn.input-to-n", -0.1},
          {"lcnn.act-steepness", -1.1},
          {"lcnn.memory-prob", -0.1},
          {"lcnn.sigma-memory", -0.1},
          {"lcnn.mu-memory", -0.1},
          {"lcnn.lms-mu", -0.1},
          {"lcnn.adapt.learning-rate", -0.1},
          {"lcnn.adapt.weight-leakage", -0.1},
          {"lcnn.adapt.abs-target-activation", -0.1},
        };
        for (int i = 0; i < (int)bench_->input_names().size(); ++i) {
            params.insert({"lcnn.mu-in-weight-" + std::to_string(i), -1.1});
            params.insert({"lcnn.sigma-in-weight-" + std::to_string(i), -0.1});
        }
        for (int i = 0; i < (int)bench_->output_names().size(); ++i) {
            params.insert({"lcnn.mu-fb-weight-" + std::to_string(i), -1.1});
            params.insert({"lcnn.sigma-fb-weight-" + std::to_string(i), -0.1});
        }
        return params;
    }

    std::map<std::string, double> named_param_ubounds() const override
    {
        std::map<std::string, double> params = {
          {"lcnn.sigma-res", 1.1},
          {"lcnn.mu-res", 1.1},
          {"lcnn.sparsity", 1.1},
          {"lcnn.in-fb-sparsity", 1.1},
          {"lcnn.leakage", 2.1},
          {"lcnn.noise", 2.0},
          {"lcnn.sigma-b", 2.0},
          {"lcnn.mu-b", 1.1},
          {"lcnn.n-state-predictors", 1.1},
          {"lcnn.train-valid-ratio", 1.1},
          {"lcnn.l2", 2.0},
          {"lcnn.enet-lambda", 2.0},
          {"lcnn.enet-alpha", 1.1},
          {"lcnn.input-to-n", 1.1},
          {"lcnn.act-steepness", 1.1},
          {"lcnn.memory-prob", 1.1},
          {"lcnn.sigma-memory", 1.1},
          {"lcnn.mu-memory", 2.1},
          {"lcnn.lms-mu", 2.0},
          {"lcnn.adapt.learning-rate", 2.0},
          {"lcnn.adapt.weight-leakage", 2.0},
          {"lcnn.adapt.abs-target-activation", 1.1},
        };
        for (int i = 0; i < (int)bench_->input_names().size(); ++i) {
            params.insert({"lcnn.mu-in-weight-" + std::to_string(i), 1.1});
            params.insert({"lcnn.sigma-in-weight-" + std::to_string(i), 2.0});
        }
        for (int i = 0; i < (int)bench_->output_names().size(); ++i) {
            params.insert({"lcnn.mu-fb-weight-" + std::to_string(i), 1.1});
            params.insert({"lcnn.sigma-fb-weight-" + std::to_string(i), 2.0});
        }
        return params;
    }

    std::vector<std::set<std::string>> param_groups() const override
    {
        if (!in_fb_group_) return {};

        std::set<std::string> mu_in_group;
        std::set<std::string> sigma_in_group;
        for (int i = 0; i < (int)bench_->input_names().size(); ++i) {
            mu_in_group.insert("lcnn.mu-in-weight-" + std::to_string(i));
            sigma_in_group.insert("lcnn.sigma-in-weight-" + std::to_string(i));
        }
        std::set<std::string> mu_fb_group;
        std::set<std::string> sigma_fb_group;
        for (int i = 0; i < (int)bench_->output_names().size(); ++i) {
            mu_fb_group.insert("lcnn.mu-fb-weight-" + std::to_string(i));
            sigma_fb_group.insert("lcnn.sigma-fb-weight-" + std::to_string(i));
        }
        return {mu_fb_group, sigma_fb_group, mu_in_group, sigma_in_group};
    }
};

class lcnn_ensemble_optimizer : public lcnn_optimizer {
public:
    using lcnn_optimizer::lcnn_optimizer;

    std::unique_ptr<net_base>
    make_net(const std::map<std::string, double>& params, prng_t& prng) const override
    {
        af::setDevice(af_device_);
        po::variables_map cfg = to_variables_map(params);
        return std::make_unique<lcnn_ensemble<>>(
          random_lcnn_ensemble(bench_->input_names(), bench_->output_names(), cfg, prng));
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
    make_net(const std::map<std::string, double>& params, prng_t& prng) const override
    {
        af::setDevice(af_device_);
        po::variables_map cfg = to_variables_map(params);
        return std::make_unique<simple_esn<>>(
          random_esn(bench_->input_names(), bench_->output_names(), cfg, prng));
    }

    std::set<std::string> available_params() const override
    {
        std::set<std::string> params = {
          "esn.sigma-res", "esn.mu-res", "esn.sparsity", "esn.leakage"};
        for (int i = 0; i < (int)bench_->input_names().size(); ++i)
            params.insert("esn.in-weight-" + std::to_string(i));
        for (int i = 0; i < (int)bench_->output_names().size(); ++i)
            params.insert("esn.fb-weight-" + std::to_string(i));
        return params;
    }

    std::map<std::string, double> named_param_x0() const override
    {
        std::map<std::string, double> params = {
          {"esn.sigma-res", 0.12},
          {"esn.mu-res", 0.0},
          {"esn.sparsity", 0.5},
          {"esn.leakage", 0.9}};
        for (int i = 0; i < (int)bench_->input_names().size(); ++i)
            params.insert({"esn.in-weight-" + std::to_string(i), 0.1});
        for (int i = 0; i < (int)bench_->output_names().size(); ++i)
            params.insert({"esn.fb-weight-" + std::to_string(i), 0.0});
        return params;
    }

    std::map<std::string, double> named_param_sigmas() const override
    {
        std::map<std::string, double> params = {
          {"esn.sigma-res", 0.01},
          {"esn.mu-res", 0.01},
          {"esn.sparsity", 0.01},
          {"esn.leakage", 0.01}};
        for (int i = 0; i < (int)bench_->input_names().size(); ++i)
            params.insert({"esn.in-weight-" + std::to_string(i), 0.01});
        for (int i = 0; i < (int)bench_->output_names().size(); ++i)
            params.insert({"esn.fb-weight-" + std::to_string(i), 0.01});
        return params;
    }

    std::map<std::string, double> named_param_lbounds() const override
    {
        std::map<std::string, double> params = {
          {"esn.sigma-res", -1.1},
          {"esn.mu-res", -1.1},
          {"esn.sparsity", -1.1},
          {"esn.leakage", -1.1}};
        for (int i = 0; i < (int)bench_->input_names().size(); ++i)
            params.insert({"esn.in-weight-" + std::to_string(i), -1.1});
        for (int i = 0; i < (int)bench_->output_names().size(); ++i)
            params.insert({"esn.fb-weight-" + std::to_string(i), -1.1});
        return params;
    }

    std::map<std::string, double> named_param_ubounds() const override
    {
        std::map<std::string, double> params = {
          {"esn.sigma-res", 1.1}, {"esn.mu-res", 1.1}, {"esn.sparsity", 1.1}, {"esn.leakage", 1.1}};
        for (int i = 0; i < (int)bench_->input_names().size(); ++i)
            params.insert({"esn.in-weight-" + std::to_string(i), 1.1});
        for (int i = 0; i < (int)bench_->output_names().size(); ++i)
            params.insert({"esn.fb-weight-" + std::to_string(i), 1.1});
        return params;
    }

    std::vector<std::set<std::string>> param_groups() const override
    {
        if (!in_fb_group_) return {};

        std::set<std::string> fb_group;
        for (int i = 0; i < (int)bench_->output_names().size(); ++i)
            fb_group.insert("esn.fb-weight-" + std::to_string(i));
        return {fb_group};
    }
};

inline po::options_description optimizer_arg_description()
{
    po::options_description optimizer_arg_desc{"Optimizer options"};
    optimizer_arg_desc.add_options()                                                        //
      ("opt.max-fevals", po::value<int>()->default_value(2000),                             //
       "Set the maximum number of evaluations of the objective function "                   //
       "aggregated evals are considered as one).")                                          //
      ("opt.lambda", po::value<int>()->default_value(25),                                   //
       "The number of offspring sampled in each step. Use -1 for automatic deduction.")     //
      ("opt.sigma", po::value<double>()->default_value(1),                                  //
       "The initial sigma value, i.e., the level of exploration. Beware that internally, "  //
       "the optimized range is [0, 10].")                                                   //
      ("opt.uncertainty", po::value<bool>()->default_value(false),                          //
       "Set up uncertainty handling.")                                                      //
      ("opt.tpa", po::value<bool>()->default_value(false),                                  //
       "Set up two-point adaptation for CMA.")                                              //
      ("opt.noisy", po::value<bool>()->default_value(false),                                //
       "Use noisy settings for CMA.")                                                       //
      ("opt.n-evals", po::value<int>()->default_value(1),                                   //
       "Run the evaluation function multiple times and aggregate those by f-value-agg.")    //
      ("opt.error-measure", po::value<std::string>()->default_value("mse"),                 //
       "The error measure used as f-value.")                                                //
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
      ("opt.param-stages-json", po::value<std::string>(),                                   //
       "A json file with parameter overrides based on evolution progress.")                 //
      ("opt.weight-cutoff", po::value<double>()->default_value(0.),                         //
       "Input and feedback weight sigmas will be cut off under this (abs) value.")          //
      ("opt.reseed-every-epoch", po::value<bool>()->default_value(false),                   //
       "Reseed the random generator in every epoch (i.e., new network in every epoch)")     //
      ("opt.multithreading", po::value<bool>()->default_value(false),                       //
       "Evaluate the individuals in the population in parallel.")                           //
      ("opt.exclude-params",                                                                //
       po::value<std::vector<std::string>>()->multitoken()->default_value(                  //
         DEFAULT_EXCLUDED_PARAMS, DEFAULT_EXCLUDED_PARAMS_STR),                             //
       "The list of parameters that should be excluded from optimization.")                 //
      ("opt.include-params",                                                                //
       po::value<std::vector<std::string>>()->multitoken(),                                 //
       "The list of parameters that should be included even if they have been excluded.")   //
      ("opt.x0-from-params", po::bool_switch(),                                             //
       "Start optimization from input arguments.")                                          //
      ("opt.in-fb-group", po::bool_switch(),                                                //
       "Optimize all input weights as a single parameter. The same for feedback.");         //
    return optimizer_arg_desc;
}

inline std::unique_ptr<net_optimizer> make_optimizer(
  benchmark_factory_t bench_factory,
  const po::variables_map& args,
  prng_t prng,
  fs::path output_dir)
{
    if (args.at("gen.net-type").as<std::string>() == "lcnn") {
        if (args.at("gen.optimizer-type").as<std::string>() == "lcnn") {
            return std::make_unique<lcnn_optimizer>(
              args, std::move(bench_factory), std::move(prng), std::move(output_dir));
        }
        throw std::invalid_argument{"Unknown lcnn optimizer type."};
    }
    if (args.at("gen.net-type").as<std::string>() == "lcnn-ensemble") {
        if (args.at("gen.optimizer-type").as<std::string>() == "lcnn-ensemble") {
            return std::make_unique<lcnn_ensemble_optimizer>(
              args, std::move(bench_factory), std::move(prng), std::move(output_dir));
        }
        throw std::invalid_argument{"Unknown lcnn optimizer type."};
    }
    if (args.at("gen.net-type").as<std::string>() == "simple-esn") {
        return std::make_unique<esn_optimizer>(
          args, std::move(bench_factory), std::move(prng), std::move(output_dir));
    }
    throw std::runtime_error{
      "Unknown net type \"" + args.at("gen.net-type").as<std::string>() + "\".\n"};
}

}  // namespace esn
