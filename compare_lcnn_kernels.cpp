// Comparison of different LCNN kernel sizes. //

#include "benchmarks.hpp"
#include "common.hpp"
#include "lcnn.hpp"
#include "optimize.hpp"

#include <boost/program_options.hpp>
#include <filesystem>
#include <iostream>

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
       "The type of the optimizer (e.g., lcnn).")                                          //
      ("gen.benchmark-set", po::value<std::string>()->default_value("narma-memory"),       //
       "Benchmark set to be evaluated.")                                                   //
      ("gen.output-dir",                                                                   //
       po::value<std::string>()->default_value("./log/lcnn_kernel_comparison/"),           //
       "Directory to store the results.")                                                  //
      ("gen.n-runs", po::value<long>()->default_value(5),                                  //
       "The number of full optimization runs of one kernel size setting.")                 //
      ("gen.kernel-sizes", po::value<std::vector<long>>()->multitoken(),                   //
       "The sizes of the kernel to be tested.")                                            //
      ("gen.state-heights", po::value<std::vector<long>>()->multitoken(),                  //
       "The heights of the reservoir to be tested.")                                       //
      ("gen.state-widths", po::value<std::vector<long>>()->multitoken(),                   //
       "The corresponding widths of the reservoir to be tested.")                          //
      ("gen.n-trials", po::value<long>()->default_value(100),                              //
       "The number of evaluations of the best network. "                                   //
       "Also the number of lines in CSV for each optimized kernel size.")                  //
      ("gen.task-offset", po::value<long>()->default_value(0),                             //
       "Cluster experiment parameter. Start evaluation from task with this index.")        //
      ("gen.n-tasks", po::value<long>()->default_value(std::numeric_limits<long>::max()),  //
       "Cluster experiment parameter. Only do this up to this number of tasks.")           //
      ("gen.overwrite", po::bool_switch(),                                                 //
       "Overwrite existing files.")                                                        //
      ("gen.af-device", po::value<int>()->default_value(0),                                //
       "ArrayFire device to be used.")                                                     //
      ("gen.seed", po::value<long>()->default_value(DEFAULT_SEED),                         //
       "Seed value for random generator. Use 0 for random_device().");                     //
    arg_desc.add(esn::benchmark_arg_description());
    arg_desc.add(esn::lcnn_arg_description());
    arg_desc.add(esn::optimizer_arg_description());

    // Parse arguments.
    po::variables_map args;
    po::store(po::parse_command_line(argc, argv, arg_desc), args);
    po::notify(args);
    if (args.count("help")) {
        std::cout << arg_desc << "\n";
        std::exit(1);
    }

    af::setDevice(args.at("gen.af-device").as<int>());
    af::info();
    std::cout << std::endl;

    long task_offset = args.at("gen.task-offset").as<long>();
    long n_tasks = args.at("gen.n-tasks").as<long>();
    fs::path output_dir = args.at("gen.output-dir").as<std::string>();
    fs::path output_file;
    if (task_offset == 0 && n_tasks == std::numeric_limits<long>::max())
        output_file = output_dir / "kernel_comparison.csv";
    else
        output_file = output_dir
          / ("kernel_comparison_" + std::to_string(task_offset) + "_" + std::to_string(n_tasks)
             + ".csv");
    if (!args.contains("overwrite") && fs::exists(output_file)) {
        std::cout << "Output file `" << output_file << "` exists, will not overwrite." << std::endl;
        return 1;
    }
    fs::create_directories(output_dir);
    std::ofstream fout{output_file};
    std::vector<std::string> param_names = {
      "run",
      "kernel-size",
      "trial",
      "f-value",
      "lcnn.state-height",
      "lcnn.state-width",
      "lcnn.kernel-height",
      "lcnn.kernel-width",
      "lcnn.topology",
      "lcnn.sigma-res",
      "lcnn.n-state-predictors",
      "lcnn.train-valid-ratio",
      "lcnn.act-steepness",
      "lcnn.input-to-n",
      "lcnn.intermediate-steps",
      "lcnn.train-aggregation",
      "lcnn.l2",
      "lcnn.enet-lambda",
      "lcnn.enet-alpha",
      "lcnn.mu-res",
      "lcnn.mu-in-weight",
      "lcnn.sigma-in-weight",
      "lcnn.mu-fb-weight",
      "lcnn.sigma-fb-weight",
      "lcnn.sparsity",
      "lcnn.leakage",
      "lcnn.noise",
      "lcnn.sigma-b",
      "lcnn.mu-b",
      "lcnn.sigma-memory-length",
      "lcnn.mu-memory-length",
      "lcnn.memory-prob",
      "lcnn.sigma-memory",
      "lcnn.mu-memory",
      "lcnn.adapt.learning-rate",
      "lcnn.adapt.weight-leakage",
      "lcnn.adapt.abs-target-activation",
    };
    fout << (rgv::join(param_names, ',') | rg::to<std::string>()) << std::endl;

    if (!args.contains("gen.state-heights"))
        throw std::invalid_argument{"lcnn.state-heights not set."};
    if (!args.contains("gen.state-widths"))
        throw std::invalid_argument{"lcnn.state-widths not set."};
    if (!args.contains("gen.kernel-sizes"))
        throw std::invalid_argument{"lcnn.kernel-sizes not set."};
    long n_sizes = args.at("gen.state-heights").as<std::vector<long>>().size();
    if (n_sizes != (long)args.at("gen.state-widths").as<std::vector<long>>().size())
        throw std::invalid_argument{"State heights and widths arguments have different length."};

    long task = -1;
    for (long run = 0; run < args.at("gen.n-runs").as<long>(); ++run) {
        for (long isize = 0; isize < n_sizes; ++isize) {
            for (long kernel_size : args.at("gen.kernel-sizes").as<std::vector<long>>()) {
                // Check if we are in a valid task range.
                ++task;
                if (task < task_offset) continue;
                if (task >= task_offset + n_tasks) continue;
                // Prepare parameters.
                long seed = esn::set_global_seed(args.at("gen.seed").as<long>() + task);
                std::cout << "Random seed: " << seed << std::endl;
                long state_height = args.at("gen.state-heights").as<std::vector<long>>().at(isize);
                long state_width = args.at("gen.state-widths").as<std::vector<long>>().at(isize);
                std::string state_size_str =
                  std::to_string(state_height) + "x" + std::to_string(state_width);
                std::cout << "Evaluating parameters:\n";
                std::cout << "task: " << task << std::endl;
                std::cout << "state size: " << state_size_str << "\n";
                std::cout << "kernel size: " << kernel_size << std::endl;
                std::cout << std::endl;
                args.insert_or_assign("lcnn.state-height", po::variable_value{state_height, false});
                args.insert_or_assign("lcnn.state-width", po::variable_value{state_width, false});
                args.insert_or_assign("lcnn.kernel-height", po::variable_value{kernel_size, false});
                args.insert_or_assign("lcnn.kernel-width", po::variable_value{kernel_size, false});
                // Store cmaes fplot data to a separate file for each run.
                std::string run_output_dir = output_dir
                  / (state_size_str + "-k" + std::to_string(kernel_size) + "-run"
                     + std::to_string(run));
                auto opt = std::make_unique<lcnn_optimizer>(
                  args, esn::make_benchmark, esn::global_prng, run_output_dir);
                cma::CMASolutions cmasols = opt->optimize();
                dVec mean = cmasols.xmean();
                po::variables_map params = opt->to_variables_map(opt->pheno_candidate(mean));
                reseed(global_prng, 13);
                for (long trial = 0; trial < args.at("gen.n-trials").as<long>(); ++trial) {
                    double f_value = opt->f_value(mean, esn::global_prng);
                    std::cout << "Best f-value: " << f_value << std::endl << std::endl;
                    for (const std::string& param : param_names) {
                        if (param == "run") {
                            fout << run;
                        } else if (param == "trial") {
                            fout << trial;
                        } else if (param == "kernel-size") {
                            fout << kernel_size;
                        } else if (param == "f-value") {
                            fout << std::setprecision(std::numeric_limits<double>::max_digits10)
                                 << f_value;
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
                            const std::vector<double>& vec =
                              params.at(param).as<std::vector<double>>();
                            for (auto it = vec.begin(); it != vec.end(); ++it) {
                                if (it != vec.begin()) fout << " ";
                                fout << std::setprecision(std::numeric_limits<double>::max_digits10)
                                     << *it;
                            }
                        } else {
                            double value = params.at(param).as<double>();
                            fout << std::setprecision(std::numeric_limits<double>::max_digits10)
                                 << value;
                        }
                        if (param != param_names.back()) fout << ",";
                    }
                    fout << std::endl;
                }
            }
        }
    }
}
