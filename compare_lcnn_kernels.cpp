// Comparison of different LCNN kernel sizes. //

#include "argument_utils.hpp"
#include "benchmarks.hpp"
#include "optimize.hpp"

#include <boost/program_options.hpp>
#include <iostream>
#include <regex>

using namespace esn;

int main(int argc, char* argv[])
{
    po::options_description arg_desc{"Generic options"};
    arg_desc.add_options()                                                            //
      ("help",                                                                        //
       "Produce help message.")                                                       //
      ("gen.net-type", po::value<std::string>()->default_value("lcnn"),               //
       "Network type, one of {simple-esn, lcnn}.")                                    //
      ("gen.optimizer-type", po::value<std::string>()->default_value("lcnn"),         //
       "The type of the optimizer (e.g., lcnn).")                                     //
      ("gen.benchmark-set", po::value<std::string>()->default_value("narma-memory"),  //
       "Benchmark set to be evaluated.")                                              //
      ("gen.output-csv",                                                              //
       po::value<std::string>()->default_value("./log/lcnn_kernels_comparison.csv"),  //
       "Output csv file with the results.")                                           //
      ("gen.n-runs", po::value<long>()->default_value(10),                            //
       "The number of full optimization runs of one kernel size setting.")            //
      ("gen.kernel-sizes", po::value<std::vector<long>>()->multitoken(),              //
       "The sizes of the kernel to be tested.")                                       //
      ("gen.n-trials", po::value<long>()->default_value(100),                         //
       "The number of evaluations of the best network. "                              //
       "Also the number of lines in CSV for each optimized kernel size.")             //
      ("gen.af-device", po::value<int>()->default_value(0),                           //
       "ArrayFire device to be used.");                                               //
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

    std::ofstream fout{args.at("gen.output-csv").as<std::string>()};
    std::vector<std::string> param_names = {"run",           "kernel-size",    "trial",
                                            "f-value",       "lcnn.topology",  "lcnn.sigma-res",
                                            "lcnn.mu-res",   "lcnn.in-weight", "lcnn.fb-weight",
                                            "lcnn.sparsity", "lcnn.leakage",   "lcnn.noise",
                                            "lcnn.sigma-b",  "lcnn.mu-b"};
    fout << (rgv::join(param_names, ',') | rg::to<std::string>()) << std::endl;
    std::string cmaes_fplot = args.at("opt.cmaes-fplot").as<std::string>();
    for (long run = 0; run < args.at("gen.n-runs").as<long>(); ++run) {
        // Store cmaes fplot data to a separate file for each run.
        std::string cmaes_fplot_run =
          std::regex_replace(cmaes_fplot, std::regex("@RUN@"), std::to_string(run));
        for (long kernel_size : args.at("gen.kernel-sizes").as<std::vector<long>>()) {
            std::cout << "Evaluating kernel size " << kernel_size << "." << std::endl;
            std::cout << std::endl;
            args.insert_or_assign("lcnn.kernel-height", po::variable_value{kernel_size, false});
            args.insert_or_assign("lcnn.kernel-width", po::variable_value{kernel_size, false});
            args.insert_or_assign("opt.cmaes-fplot", po::variable_value{cmaes_fplot_run, false});
            auto opt =
              std::make_unique<lcnn_optimizer>(args, esn::make_benchmark(args), esn::global_prng);
            cma::CMASolutions cmasols = opt->optimize();
            dVec mean = cmasols.xmean();
            po::variables_map params = opt->to_variables_map(opt->pheno_candidate(mean));
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
                    } else if (param == "lcnn.topology") {
                        fout << args.at("lcnn.topology").as<std::string>();
                    } else if (param == "f-value") {
                        fout << std::setprecision(std::numeric_limits<double>::max_digits10)
                             << f_value;
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
