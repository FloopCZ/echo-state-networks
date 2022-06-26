// Echo state network usage example. //

#include "visual.hpp"

#include "analysis.hpp"
#include "argument_utils.hpp"
#include "benchmarks.hpp"
#include "lcnn.hpp"
#include "simple_esn.hpp"

#include <chrono>
#include <fstream>
#include <iostream>
#include <thread>

namespace po = boost::program_options;

int main(int argc, char* argv[])
{
    po::options_description arg_desc{"Generic options"};
    arg_desc.add_options()                                                       //
      ("help",                                                                   //
       "Produce help message.")                                                  //
      ("gen.net-type", po::value<std::string>()->default_value("lcnn"),          //
       "Network type, one of {simple-esn, lcnn}.")                               //
      ("gen.benchmark-set", po::value<std::string>()->default_value("narma10"),  //
       "Benchmark set to be evaluated.")                                         //
      ("gen.plot-size", po::value<long>()->default_value(800),                   //
       "The size (i.e., the height and the width) of each plot.")                //
      ("gen.sleep", po::value<long>()->default_value(0),                         //
       "The number of milliseconds to sleep between steps.")                     //
      ("gen.history", po::value<long>()->default_value(500),                     //
       "The length of the plot history.")                                        //
      ("gen.csv-out", po::value<std::string>()->default_value("visual.csv"),     //
       "CSV file where to save the inputs, outputs and other statistics.")       //
      ("gen.skip", po::value<long>()->default_value(0),                          //
       "Do not plot until this number of steps has passed.")                     //
      ("gen.af-device", po::value<int>()->default_value(0),                      //
       "ArrayFire device to be used.");                                          //
    arg_desc.add(esn::benchmark_arg_description());
    po::variables_map args = esn::parse_conditional(
      argc, argv, arg_desc,
      {{"gen.net-type",                                   //
        {{"lcnn", esn::lcnn_arg_description()},           //
         {"simple-esn", esn::esn_arg_description()}}}});  //

    af::setDevice(args.at("gen.af-device").as<int>());
    af::info();
    std::cout << std::endl;

    std::unique_ptr<esn::benchmark_set_base> bench = esn::make_benchmark(args);
    std::unique_ptr<esn::net_base> net =
      esn::make_net(bench->n_ins(), bench->n_outs(), args, esn::global_prng);

    esn::visualizer plt{
      args.at("gen.sleep").as<long>(), args.at("gen.history").as<long>(),
      args.at("gen.plot-size").as<long>(), args.at("gen.skip").as<long>()};
    plt.register_callback(*net);

    esn::file_saver csv_writer;
    if (args.contains("gen.csv-out")) {
        std::string csv_out = args.at("gen.csv-out").as<std::string>();
        csv_writer = esn::file_saver{csv_out};
        csv_writer.register_callback(*net);
    }

    bench->evaluate(*net, esn::global_prng);
    plt.wait_for_close();
    return 0;
}
