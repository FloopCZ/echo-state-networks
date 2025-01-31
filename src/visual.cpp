// Echo state network usage example. //

#include "visual.hpp"

#include "argument_utils.hpp"
#include "benchmarks.hpp"
#include "lcnn.hpp"
#include "lcnn_fixer.hpp"
#include "simple_esn.hpp"

#include <filesystem>
#include <iostream>

namespace po = boost::program_options;
namespace fs = std::filesystem;

int main(int argc, char* argv[])
{
    po::options_description arg_desc{"Generic options"};
    arg_desc.add_options()                                                        //
      ("help",                                                                    //
       "Produce help message.")                                                   //
      ("gen.net-type", po::value<std::string>()->default_value("lcnn"),           //
       "Network type, one of {simple-esn, lcnn}.")                                //
      ("gen.benchmark-set", po::value<std::string>()->default_value("narma10"),   //
       "Benchmark set to be evaluated.")                                          //
      ("gen.plot-size", po::value<long>()->default_value(800),                    //
       "The size (i.e., the height and the width) of each plot.")                 //
      ("gen.sleep", po::value<long>()->default_value(0),                          //
       "The number of milliseconds to sleep between steps.")                      //
      ("gen.history", po::value<long>()->default_value(500),                      //
       "The length of the plot history.")                                         //
      ("gen.output-dir", po::value<std::string>()->default_value("log/visual/"),  //
       "Directory where to save the inputs, outputs and other statistics.")       //
      ("gen.skip", po::value<long>()->default_value(0),                           //
       "Do not plot until this number of steps has passed.")                      //
      ("gen.real-time-visual", po::bool_switch(),                                 //
       "Do real-time visualization.")                                             //
      ("gen.af-device", po::value<int>()->default_value(0),                       //
       "ArrayFire device to be used.")                                            //
      ("gen.seed", po::value<long>()->default_value(esn::DEFAULT_SEED),           //
       "Seed value for random generator. Use 0 for random_device().");            //
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

    std::unique_ptr<esn::benchmark_set_base> bench = esn::make_benchmark(args);
    std::unique_ptr<esn::net_base> net =
      esn::make_net(bench->input_names(), bench->output_names(), args, esn::global_prng);

    std::optional<esn::visualizer> plt;
    if (args.at("gen.real-time-visual").as<bool>()) {
        plt.emplace(
          args.at("gen.sleep").as<long>(), args.at("gen.history").as<long>(),
          args.at("gen.plot-size").as<long>(), args.at("gen.skip").as<long>());
        plt->register_callback(*net);
    }

    fs::path output_dir = args.at("gen.output-dir").as<std::string>();
    fs::create_directories(output_dir);

    esn::file_saver csv_writer{output_dir / "trace.csv"};
    csv_writer.register_callback(*net);

    bench->evaluate(*net, esn::global_prng);
    if (plt) plt->wait_for_close();
    return 0;
}
