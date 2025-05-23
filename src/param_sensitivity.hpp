#include "argument_utils.hpp"
#include "benchmark_results.hpp"
#include "benchmarks.hpp"
#include "lcnn.hpp"
#include "lcnn_fixer.hpp"
#include "simple_esn.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>

namespace po = boost::program_options;
namespace fs = std::filesystem;

template <typename T>
int param_sensitivity(int argc, char* argv[])
{
    po::options_description arg_desc{"Generic options"};
    arg_desc.add_options()                                                                     //
      ("help",                                                                                 //
       "Produce help message.")                                                                //
      ("gen.net-type", po::value<std::string>()->default_value("lcnn"),                        //
       "Network type, one of {simple-esn, lcnn}.")                                             //
      ("gen.benchmark-set", po::value<std::string>()->default_value("narma10"),                //
       "Benchmark set to be evaluated.")                                                       //
      ("gen.param", po::value<std::string>(),                                                  //
       "The parameter to be optimized.")                                                       //
      ("gen.grid-start", po::value<T>(),                                                       //
       "The grid will start at the origingal value plus grid-start.")                          //
      ("gen.grid-step", po::value<T>(),                                                        //
       "The size of the grid step.")                                                           //
      ("gen.grid-stop", po::value<T>(),                                                        //
       "The grid will end at the orignal value plus grid-stop.")                               //
      ("gen.output-dir", po::value<std::string>()->default_value("./log/param_sensitivity/"),  //
       "Output directory file with the results.")                                              //
      ("gen.af-device", po::value<int>()->default_value(0),                                    //
       "ArrayFire device to be used.")                                                         //
      ("gen.seed", po::value<long>()->default_value(esn::DEFAULT_SEED),                        //
       "Seed value for random generator. Use 0 for random_device().");                         //
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
    fs::create_directories(output_dir);
    std::ofstream fout{output_dir / "sensitivity_grid.csv"};
    std::string net_type = args.at("gen.net-type").as<std::string>();
    T param_orig = args.at(args.at("gen.param").as<std::string>()).as<T>();
    T grid_start = args.at("gen.grid-start").as<T>();
    T grid_step = args.at("gen.grid-step").as<T>();
    T grid_stop = args.at("gen.grid-stop").as<T>();

    fout << args.at("gen.param").as<std::string>() << ",f-value" << std::endl;
    for (T grid_delta = grid_start; grid_delta <= grid_stop; grid_delta += grid_step) {
        T param = param_orig + grid_delta;
        args.insert_or_assign(
          args.at("gen.param").as<std::string>(), po::variable_value{param, false});
        auto bench = esn::make_benchmark(args);
        auto net =
          esn::make_net(bench->input_names(), bench->output_names(), args, esn::global_prng);
        esn::benchmark_results results = bench->evaluate(*net, esn::global_prng);
        fout << param << "," << results.at("mse").mean() << std::endl;
    }

    return 0;
}
