// Echo state network evaluation. //

#include "argument_utils.hpp"
#include "benchmarks.hpp"
#include "lcnn.hpp"
#include "lcnn_ensemble.hpp"
#include "simple_esn.hpp"

#include <iostream>

namespace po = boost::program_options;

/// Evaluate the net on the given benchmark.
///
/// \param net_factory The network to be tested.
/// \param n_evals The number of complete reevaluations of the provided net.
template <typename NetFactory>
std::vector<double>
evaluate(NetFactory net_factory, std::unique_ptr<esn::benchmark_set_base> bench, long n_evals)
{
    int af_device = af::getDevice();
    // Evaluate the individual repeats in parallel.
    std::vector<double> results(n_evals);
    std::for_each(std::execution::seq, results.begin(), results.end(), [&](double& r) {
        // We need to make sure the device is set properly, otherwise
        // it sometimes fails on XID errors.
        af::setDevice(af_device);
        auto net = net_factory(bench->input_names(), bench->output_names());
        r = bench->evaluate(*net, esn::global_prng);
    });
    return results;
}

int main(int argc, char* argv[])
{
    po::options_description arg_desc{"Generic options"};
    arg_desc.add_options()                                                         //
      ("help",                                                                     //
       "Produce help message.")                                                    //
      ("gen.net-type", po::value<std::string>()->default_value("lcnn"),            //
       "Network type, one of {simple-esn, lcnn}.")                                 //
      ("gen.benchmark-set", po::value<std::string>()->default_value("narma10"),    //
       "Benchmark set to be evaluated.")                                           //
      ("gen.n-evals", po::value<long>()->default_value(3),                         //
       "The number of complete reevaluations of the provided set of parameters.")  //
      ("gen.af-device", po::value<int>()->default_value(0),                        //
       "ArrayFire device to be used.")                                             //
      ("gen.seed", po::value<long>()->default_value(esn::DEFAULT_SEED),            //
       "Seed value for random generator. Use 0 for random_device().");             //
    arg_desc.add(esn::benchmark_arg_description());
    po::variables_map args = esn::parse_conditional(
      argc, argv, arg_desc,
      {{"gen.net-type",                                            //
        {{"lcnn", esn::lcnn_arg_description()},                    //
         {"lcnn-ensemble", esn::lcnn_ensemble_arg_description()},  //
         {"simple-esn", esn::esn_arg_description()}}}});           //

    long seed = args.at("gen.seed").as<long>();
    if (seed != 0) esn::global_prng.seed(seed);

    af::setDevice(args.at("gen.af-device").as<int>());
    af::info();
    std::cout << std::endl;

    std::unique_ptr<esn::benchmark_set_base> bench = esn::make_benchmark(args);
    auto net_factory = [&](auto... fwd) { return esn::make_net(fwd..., args, esn::global_prng); };
    std::string name = args.at("gen.benchmark-set").as<std::string>() + " "
      + args.at("bench.error-measure").as<std::string>();
    af::timer::start();
    esn::benchmark_results results;
    results.insert(
      name, evaluate(net_factory, std::move(bench), args.at("gen.n-evals").as<long>()));
    std::cout << results << std::endl;
    std::cout << "elapsed time: " << af::timer::stop() << std::endl;

    return 0;
}
