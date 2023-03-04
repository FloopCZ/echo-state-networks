// Echo state network optimization. //

#include "optimize.hpp"

#include "argument_utils.hpp"
#include "benchmarks.hpp"

#include <iostream>

using namespace esn;

int main(int argc, char* argv[])
{
    po::options_description arg_desc{"Generic options"};
    arg_desc.add_options()                                                         //
      ("help",                                                                     //
       "Produce help message.")                                                    //
      ("gen.net-type", po::value<std::string>()->default_value("lcnn"),            //
       "Network type, one of {simple-esn, lcnn}.")                                 //
      ("gen.optimizer-type", po::value<std::string>()->default_value("lcnn"),      //
       "The type of the optimizer (e.g., lcnn, simple-esn).")                      //
      ("gen.benchmark-set", po::value<std::string>()->default_value("narma10"),    //
       "Benchmark set to be evaluated.")                                           //
      ("gen.output-csv",                                                           //
       po::value<std::string>()->default_value("./log/optimization_results.csv"),  //
       "Output csv file with the results.")                                        //
      ("gen.n-runs", po::value<long>()->default_value(10),                         //
       "The number of full optimization runs of the provided set of parameters.")  //
      ("gen.n-trials", po::value<long>()->default_value(100),                      //
       "The number of evaluations of the best network. "                           //
       "The number of lines in CSV is n-runs * n-trials.")                         //
      ("gen.af-device", po::value<int>()->default_value(0),                        //
       "ArrayFire device to be used.");                                            //
    arg_desc.add(esn::benchmark_arg_description());
    arg_desc.add(esn::optimizer_arg_description());
    po::variables_map args = esn::parse_conditional(
      argc, argv, arg_desc,
      {{"gen.net-type",                                   //
        {{"lcnn", esn::lcnn_arg_description()},           //
         {"simple-esn", esn::esn_arg_description()}}}});  //

    af::setDevice(args.at("gen.af-device").as<int>());
    af::info();
    std::cout << std::endl;

    std::ofstream fout{args.at("gen.output-csv").as<std::string>()};
    std::string net_type = args.at("gen.net-type").as<std::string>();
    std::vector<std::string> param_names = {
      "run",
      "trial",
      "f-value",
      net_type + ".topology",
      net_type + ".sigma-res",
      net_type + ".mu-res",
      net_type + ".in-weight",
      net_type + ".fb-weight",
      net_type + ".sparsity",
      net_type + ".leakage",
      net_type + ".noise",
      net_type + ".sigma-b",
      net_type + ".mu-b"};
    fout << (rgv::join(param_names, ',') | rg::to<std::string>()) << std::endl;
    std::string cmaes_fplot = args.at("opt.cmaes-fplot").as<std::string>();

    for (long run = 0; run < args.at("gen.n-runs").as<long>(); ++run) {
        std::cout << "Run " << run << std::endl;

        // Store cmaes fplot data to a separate file for each run.
        std::string cmaes_fplot_run =
          std::regex_replace(cmaes_fplot, std::regex("@RUN@"), std::to_string(run));
        args.insert_or_assign("opt.cmaes-fplot", po::variable_value{cmaes_fplot_run, false});

        std::unique_ptr<esn::benchmark_set_base> bench = esn::make_benchmark(args);
        std::unique_ptr<esn::net_optimizer> opt =
          esn::make_optimizer(std::move(bench), args, global_prng);
        cma::CMASolutions cmasols = opt->optimize();
        cma::Candidate best_candidate = cmasols.get_best_seen_candidate();
        std::vector<double> best_pheno_candidate = opt->pheno_candidate(best_candidate);
        std::cout << "Best seen candidate:\n";
        opt->print_candidate(std::cout, best_candidate) << std::endl;
        std::cout << "Distribution mean:\n";
        opt->print_candidate(std::cout, cmasols.xmean(), global_prng) << std::endl;

        // CSV rows
        esn::net_evaluation_result_t best_evaluation = std::move(opt->best_evaluation());
        po::variables_map params = opt->to_variables_map(best_evaluation.params);
        for (long trial = 0; trial < args.at("gen.n-trials").as<long>(); ++trial) {
            for (const std::string& param : param_names) {
                if (param == "run") {
                    fout << run;
                } else if (param == "trial") {
                    fout << trial;
                } else if (param == "f-value") {
                    double f_value = opt->evaluate_net(*best_evaluation.net, global_prng);
                    fout << std::setprecision(std::numeric_limits<double>::max_digits10) << f_value;
                } else if (param == net_type + ".topology") {
                    fout << args.at(net_type + ".topology").as<std::string>();
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
