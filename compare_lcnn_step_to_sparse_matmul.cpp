#include "lcnn_step.hpp"
#include "misc.hpp"

#include <af/random.h>
#include <arrayfire.h>
#include <iostream>

struct sparse_variants {
    af::array lcnn_data;
    af::array dense;
    af::array sparse_csr;
};

sparse_variants generate_weights(
  int state_height, int state_width, int kernel_height, int kernel_width, af::randomEngine& af_prng)
{
    // generate dense reservoir weights
    af::dim4 lcnn_dims{state_height, state_width, kernel_height, kernel_width};
    af::array lcnn_data = af::randu(lcnn_dims, af::dtype::f64, af_prng);

    // generate sparse reservoir weights
    std::vector<double> dense_data(state_height * state_width * state_height * state_width);

    for (int i = 0; i < state_height; ++i) {
        for (int j = 0; j < state_width; ++j) {
            for (int k = 0; k < kernel_height; ++k) {
                for (int l = 0; l < kernel_width; ++l) {
                    int from_i = (i + k - kernel_height / 2 + state_height) % state_height;
                    int from_j = (j + l - kernel_width / 2 + state_width) % state_width;
                    int full_index = i + j * state_height
                      + (from_i + from_j * state_height) * state_height * state_width;
                    assert(full_index >= 0 && full_index < (long)dense_data.size());
                    int sparse_index = i + j * state_height + k * state_height * state_width
                      + l * state_height * state_width * kernel_height;
                    assert(sparse_index >= 0 && sparse_index < (long)lcnn_data.elements());
                    dense_data[full_index] += af::flat(lcnn_data)(sparse_index).scalar<double>();
                }
            }
        }
    }
    af::array dense{state_height * state_width, state_height * state_width, dense_data.data()};

    return {lcnn_data, dense, af::sparse(dense, AF_STORAGE_CSR)};
}

int main(int argc, char* argv[])
{
    int state_height = 80;
    int state_width = 100;
    int kernel_height = 7;
    int kernel_width = 7;
    if (argc != 1 && argc != 5) {
        std::cerr << "Usage: " << argv[0] << " state_height state_width kernel_height kernel_width"
                  << std::endl;
        return -1;
    }
    if (argc == 5) {
        state_height = std::stoi(argv[1]);
        state_width = std::stoi(argv[2]);
        kernel_height = std::stoi(argv[3]);
        kernel_width = std::stoi(argv[4]);
    }
    std::cout << "Benchmarking with state_height=" << state_height
              << ", state_width=" << state_width << ", kernel_height=" << kernel_height
              << ", kernel_width=" << kernel_width << std::endl;

    try {
        af::info();
        std::cout << std::endl;

        af::randomEngine af_prng = af::randomEngine(AF_RANDOM_ENGINE_DEFAULT, 13);
        const af::array state = af::randu({state_height, state_width}, af::dtype::f64, af_prng);
        const sparse_variants data =
          generate_weights(state_height, state_width, kernel_height, kernel_width, af_prng);

        int count = 1000;
        esn::Timer timer;

        {
            af::array dense_result = af::matmul(data.dense, af::flat(state));
            dense_result.eval();
            timer.start("dense", false, count);
            for (int i = 0; i < count; ++i) {
                dense_result = af::matmul(data.dense, af::flat(state));
                dense_result.eval();
            }
            timer.stop("dense");
        }
        {
            af::array sparse_csr_result = af::matmul(data.sparse_csr, af::flat(state));
            sparse_csr_result.eval();
            timer.start("sparse_csr", false, count);
            for (int i = 0; i < count; ++i) {
                sparse_csr_result = af::matmul(data.sparse_csr, af::flat(state));
                sparse_csr_result.eval();
            }
            timer.stop("sparse_csr");
        }
        {
            timer.start("lcnn", false, count);
            af::array lcnn_result = lcnn_step(state, data.lcnn_data);
            lcnn_result.eval();
            for (int i = 0; i < count; ++i) {
                af::array lcnn_result = lcnn_step(state, data.lcnn_data);
                lcnn_result.eval();
            }
            timer.stop("lcnn");
        }

    } catch (af::exception& e) {
        std::cerr << "Arrayfire Exception: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}