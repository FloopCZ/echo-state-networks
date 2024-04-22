// Echo state network optimization. //

#include "optimize.hpp"

#include "argument_utils.hpp"
#include "benchmarks.hpp"
#include "common.hpp"

#include <boost/algorithm/string.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace esn;

namespace fs = std::filesystem;

int main(int argc, char* argv[])
{
    po::options_description arg_desc{"Generic options"};
    arg_desc.add_options()                                                                 //
      ("help",                                                                             //
       "Produce help message.")                                                            //
      ("gen.net-type", po::value<std::string>()->default_value("lcnn"),                    //
       "Network type, one of {simple-esn, lcnn}.")                                         //
      ("gen.optimizer-type", po::value<std::string>()->default_value("lcnn"),              //
       "The type of the optimizer (e.g., lcnn, simple-esn).")                              //
      ("gen.benchmark-set", po::value<std::string>()->default_value("narma10"),            //
       "Benchmark set to be evaluated.")                                                   //
      ("gen.output-dir",                                                                   //
       po::value<std::string>()->default_value("./log/optimization/"),                     //
       "Output directory with the results.")                                               //
      ("gen.n-runs", po::value<long>()->default_value(5),                                  //
       "The number of full optimization runs of the provided set of parameters.")          //
      ("gen.n-trials", po::value<long>()->default_value(100),                              //
       "The number of evaluations of the best network. "                                   //
       "The number of lines in CSV is n-runs * n-trials.")                                 //
      ("gen.task-offset", po::value<long>()->default_value(0),                             //
       "Cluster experiment parameter. Start evaluation from task with this index.")        //
      ("gen.n-tasks", po::value<long>()->default_value(std::numeric_limits<long>::max()),  //
       "Cluster experiment parameter. Only do this up to this number of tasks.")           //
      ("gen.overwrite", po::bool_switch(),                                                 //
       "Overwrite existing files.")                                                        //
      ("gen.af-device", po::value<int>()->default_value(0),                                //
       "ArrayFire device to be used.")                                                     //
      ("gen.seed", po::value<long>()->default_value(esn::DEFAULT_SEED),                    //
       "Seed value for random generator. Use 0 for random_device().");                     //
    arg_desc.add(esn::benchmark_arg_description());
    arg_desc.add(esn::optimizer_arg_description());
    po::variables_map args = esn::parse_conditional(
      argc, argv, arg_desc,
      {{"gen.net-type",                                            //
        {{"lcnn", esn::lcnn_arg_description()},                    //
         {"lcnn-ensemble", esn::lcnn_ensemble_arg_description()},  //
         {"simple-esn", esn::esn_arg_description()}}}});           //

    af::setDevice(args.at("gen.af-device").as<int>());
    af::info();
    std::cout << std::endl;

    long task_offset = args.at("gen.task-offset").as<long>();
    long n_tasks = args.at("gen.n-tasks").as<long>();
    fs::path output_dir = args.at("gen.output-dir").as<std::string>();
    fs::path output_file;
    if (task_offset == 0 && n_tasks == std::numeric_limits<long>::max())
        output_file = output_dir / "optimization_results.csv";
    else
        output_file = output_dir
          / ("optimization_results_" + std::to_string(task_offset) + "_" + std::to_string(n_tasks)
             + ".csv");
    if (!args.contains("overwrite") && fs::exists(output_file)) {
        std::cout << "Output file `" << output_file << "` exists, will not overwrite." << std::endl;
        return 1;
    }
    fs::create_directories(output_dir);
    std::ofstream fout{output_file};
    std::string net_type = args.at("gen.net-type").as<std::string>();
    std::vector<std::string> param_names = {
      "run",
      "trial",
      "f-value",
      net_type + ".topology",
      net_type + ".sigma-res",
      net_type + ".mu-res",
      net_type + ".sparsity",
      net_type + ".leakage",
      net_type + ".noise",
      net_type + ".sigma-b",
      net_type + ".mu-b"};
    if (net_type == "esn") {
        param_names.push_back("esn.in-weight");
        param_names.push_back("esn.fb-weight");
    }
    if (net_type == "lcnn" || net_type == "lcnn-ensemble") {
        param_names.push_back("lcnn.in-fb-sparsity");
        param_names.push_back("lcnn.mu-fb-weight");
        param_names.push_back("lcnn.sigma-fb-weight");
        param_names.push_back("lcnn.state-height");
        param_names.push_back("lcnn.state-width");
        param_names.push_back("lcnn.kernel-height");
        param_names.push_back("lcnn.kernel-width");
        param_names.push_back("lcnn.input-to-n");
        param_names.push_back("lcnn.n-state-predictors");
        param_names.push_back("lcnn.train-valid-ratio");
        param_names.push_back("lcnn.act-steepness");
        param_names.push_back("lcnn.n-train-trials");
        param_names.push_back("lcnn.intermediate-steps");
        param_names.push_back("lcnn.train-aggregation");
        param_names.push_back("lcnn.l2");
        param_names.push_back("lcnn.enet-lambda");
        param_names.push_back("lcnn.enet-alpha");
        param_names.push_back("lcnn.memory-length");
        param_names.push_back("lcnn.memory-prob");
        param_names.push_back("lcnn.sigma-memory");
        param_names.push_back("lcnn.mu-memory");
        param_names.push_back("lcnn.adapt.learning-rate");
        param_names.push_back("lcnn.adapt.weight-leakage");
        param_names.push_back("lcnn.adapt.abs-target-activation");
    }
    if (net_type == "lcnn-ensemble") {
        param_names.push_back("lcnn-ensemble.n");
    }
    fout << boost::join(param_names, ",") << std::endl;

    long task = -1;
    for (long run = 0; run < args.at("gen.n-runs").as<long>(); ++run) {
        // Check if we are in a valid task range.
        ++task;
        if (task < task_offset) continue;
        if (task >= task_offset + n_tasks) continue;
        // Prepare parameters.
        std::cout << "Run " << run << std::endl;
        std::cout << "Task: " << task << std::endl;
        long seed = esn::set_global_seed(args.at("gen.seed").as<long>() + task);
        std::cout << "Random seed: " << seed << std::endl;
        fs::path run_output_dir = output_dir / ("run" + std::to_string(run));
        std::unique_ptr<esn::net_optimizer> opt =
          esn::make_optimizer(esn::make_benchmark, args, global_prng, run_output_dir);
        cma::CMASolutions cmasols = opt->optimize();
        std::cout << "Distribution mean:\n";
        opt->print_candidate(std::cout, cmasols.xmean(), global_prng) << std::endl;

        // CSV rows
        std::cout << "Best seen candidate:\n";
        esn::net_evaluation_result_t best_evaluation = std::move(opt->best_evaluation());
        if (best_evaluation.net == nullptr) throw std::runtime_error{"No best network."};
        opt->print_result(std::cout, best_evaluation) << std::endl;
        po::variables_map params = opt->to_variables_map(best_evaluation.params);
        fs::path param_file = run_output_dir / "best-model" / "args.txt";
        std::ofstream param_out{param_file};
        param_out << params;
        reseed(global_prng, 13);
        for (long trial = 0; trial < args.at("gen.n-trials").as<long>(); ++trial) {
            for (const std::string& param : param_names) {
                if (param == "run") {
                    fout << run;
                } else if (param == "trial") {
                    fout << trial;
                } else if (param == "f-value") {
                    best_evaluation.net->reset();
                    double f_value = opt->benchmark().evaluate(*best_evaluation.net, global_prng);
                    fout << std::setprecision(std::numeric_limits<double>::max_digits10) << f_value;
                } else if (typeid(int) == params.at(param).value().type()) {
                    fout << params.at(param).as<int>();
                } else if (typeid(long) == params.at(param).value().type()) {
                    fout << params.at(param).as<long>();
                } else if (typeid(std::string) == params.at(param).value().type()) {
                    fout << params.at(param).as<std::string>();
                } else if (typeid(std::vector<long>) == params.at(param).value().type()) {
                    const std::vector<long>& vec = params.at(param).as<std::vector<long>>();
                    for (auto it = vec.begin(); it != vec.end(); ++it) {
                        if (it != vec.begin()) fout << " ";
                        fout << *it;
                    }
                } else if (typeid(std::vector<double>) == params.at(param).value().type()) {
                    const std::vector<double>& vec = params.at(param).as<std::vector<double>>();
                    for (auto it = vec.begin(); it != vec.end(); ++it) {
                        if (it != vec.begin()) fout << " ";
                        fout << std::setprecision(std::numeric_limits<double>::max_digits10) << *it;
                    }
                } else {
                    double value = params.at(param).as<double>();
                    fout << std::setprecision(std::numeric_limits<double>::max_digits10) << value;
                }
                if (param != param_names.back()) fout << ",";
            }
            fout << std::endl;
        }
    }
}
