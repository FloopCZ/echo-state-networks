// Echo state network evaluation. //

#include "argument_utils.hpp"
#include "benchmarks.hpp"
#include "lcnn.hpp"
#include "lcnn_ensemble.hpp"
#include "lcnn_fixer.hpp"
#include "simple_esn.hpp"

#include <filesystem>
#include <iostream>

namespace po = boost::program_options;
namespace fs = std::filesystem;

/// Evaluate the net on the given benchmark.
///
/// \param net_factory The network to be tested.
/// \param n_evals The number of complete reevaluations of the provided net.
template <typename NetFactory>
esn::benchmark_results
evaluate(NetFactory net_factory, std::unique_ptr<esn::benchmark_set_base> bench, long n_evals)
{
    // Prepare benchmark result structure.
    esn::benchmark_results results;
    // Evaluate the individual repeats.
    std::vector<std::vector<double>> raw_results(n_evals);
    for (long i = 0; i < n_evals; ++i) {
        auto net = net_factory(bench->input_names(), bench->output_names());
        results.insert(bench->evaluate(*net, esn::global_prng));
    }
    return results;
}

int main(int argc, char* argv[])
{
    po::options_description arg_desc{"Generic options"};
    arg_desc.add_options()                                                          //
      ("help",                                                                      //
       "Produce help message.")                                                     //
      ("gen.net-type", po::value<std::string>()->default_value("lcnn"),             //
       "Network type, one of {simple-esn, lcnn}.")                                  //
      ("gen.benchmark-set", po::value<std::string>()->default_value("narma10"),     //
       "Benchmark set to be evaluated.")                                            //
      ("gen.n-evals", po::value<long>()->default_value(3),                          //
       "The number of complete reevaluations of the provided set of parameters.")   //
      ("gen.output-dir", po::value<std::string>()->default_value("log/evaluate/"),  //
       "Directory where to save the evaluation results.")                           //
      ("gen.overwrite", po::bool_switch(),                                          //
       "Overwrite existing files.")                                                 //
      ("gen.af-device", po::value<int>()->default_value(0),                         //
       "ArrayFire device to be used.")                                              //
      ("gen.seed", po::value<long>()->default_value(esn::DEFAULT_SEED),             //
       "Seed value for random generator. Use 0 for random_device().");              //
    arg_desc.add(esn::benchmark_arg_description());
    po::variables_map args = esn::parse_conditional(
      argc, argv, arg_desc,
      {{"gen.net-type",                                            //
        {{"lcnn", esn::lcnn_arg_description()},                    //
         {"lcnn-ensemble", esn::lcnn_ensemble_arg_description()},  //
         {"lcnn-fixer", esn::lcnn_fixer_arg_description()},        //
         {"simple-esn", esn::esn_arg_description()}}}});           //

    long seed = esn::set_global_seed(args.at("gen.seed").as<long>());
    std::cout << "Random seed: " << seed << std::endl;

    af::setDevice(args.at("gen.af-device").as<int>());
    af::info();
    std::cout << std::endl;

    fs::path output_dir = args.at("gen.output-dir").as<std::string>();
    fs::path output_csv = output_dir / "results.csv";
    if (!args.at("gen.overwrite").as<bool>() && fs::exists(output_csv)) {
        std::cout << "Output file `" << output_csv << "` exists, will not overwrite." << std::endl;
        return 1;
    }

    std::unique_ptr<esn::benchmark_set_base> bench = esn::make_benchmark(args);
    auto net_factory = [&](auto... fwd) { return esn::make_net(fwd..., args, esn::global_prng); };
    long n_evals = args.at("gen.n-evals").as<long>();

    af::timer::start();
    esn::benchmark_results results = evaluate(net_factory, std::move(bench), n_evals);
    std::cout << "Aggregated results\n" << results << std::endl;
    std::cout << "elapsed time: " << af::timer::stop() << std::endl;

    results.to_csv(output_csv, "model");

    return 0;
}
