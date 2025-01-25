#pragma once

// Various tools for echo state network analysis. //

#include "arrayfire_utils.hpp"
#include "benchmark_results.hpp"
#include "net.hpp"

#include <arrayfire.h>
#include <cassert>
#include <cmath>
#include <execution>
#include <random>
#include <range/v3/all.hpp>

namespace esn {

namespace rg = ranges;
namespace rgv = ranges::views;

/// Generate a Nonlinear AutoRegressive Moving Average of a sequence.
///
/// \param xs The sequence from which the NARMA will be calculated of size [time].
/// \param history The time to be taken into account.
/// \param coefs The coefficients of the sequence.
/// \returns The NARMA of the sequence xs of size [time].
af::array narma(const af::array& xs, long history, long tau, std::vector<double> coefs)
{
    assert(xs.numdims() == 1);
    assert(coefs.size() == 4);
    assert(tau >= 1);
    std::vector<double> xs_vec = af_utils::to_vector(xs);
    std::vector<double> ys_vec(xs.dims(0));

    // return zero before start
    auto at = [](const std::vector<double>& vec, long i) -> double {
        if (i < 0) return 0;
        return vec.at(i);
    };
    // construct the narma sequence
    for (long i = 0; i < (long)xs_vec.size(); ++i) {
        double sum = 0;
        for (long j = 1; j <= (long)history; ++j) sum += ys_vec.at(std::max(0L, i - j * tau));
        ys_vec.at(i) =                                             //
          +coefs.at(0)                                             //
            * at(ys_vec, i - tau)                                  //
          + coefs.at(1)                                            //
            * at(ys_vec, i - tau)                                  //
            * sum                                                  //
          + coefs.at(2)                                            //
            * at(xs_vec, i - history * tau) * at(xs_vec, i - tau)  //
          + coefs.at(3);                                           //
    }
    assert(rg::size(xs_vec) == rg::size(ys_vec));

    // copy the result back to af::array
    return af_utils::to_array(ys_vec).as(xs.type());
}

/// Generate a random Mackey-Glass sequence.
///
/// See Holzmann 2008: Echo State Networks with Filter Neurons and a Delay&Sum Readout.
///
/// \param len The length of the sequence to be generated.
/// \param tau The tau constant.
/// \param delta The time delta constant.
af::array mackey_glass(long len, double tau, double delta, af::dtype dtype, prng_t& prng)
{
    // the beginning is random, and it will be skipped, so generate a bit extra
    assert(1. / delta == std::ceil(1. / delta));  // assert integral 1/delta
    long dense_len = (len + tau) / delta + 1;
    std::vector<double> dense(dense_len);
    // construct the sequence
    for (long i = 0; i <= tau / delta; ++i) {
        dense.at(i) = 1.2 + std::uniform_real_distribution<>{-0.1, 0.1}(prng);
    }
    double y_t = dense.at(tau / delta);
    for (long i = tau / delta + 1; i < dense_len; ++i) {
        double y_far_t = dense.at(i - 1 - tau / delta);
        y_t += delta * (0.2 * y_far_t / (1. + std::pow(y_far_t, 10)) - 0.1 * y_t);
        dense.at(i) = y_t;
    }
    // skip the random beginning and subsample according to delta
    std::vector<double> ys = dense                  //
      | rgv::slice(long(tau / delta + 1), rg::end)  //
      | rgv::stride(long(1. / delta))               //
      | rg::to_vector;
    assert((long)ys.size() == len);
    return af_utils::to_array(ys).as(dtype);
}

/// Generate a memory matrix of a sequence.
///
/// Column `i` in the memory matrix is the original sequence delayed by `i` steps.
///
/// \param xs The sequence from which the memory matrix will be calculated of size [time].
/// \param history The time to be taken into account.
/// \returns The memory matrix of the sequence xs of size [history, time].
af::array memory_matrix(const af::array& xs, long history)
{
    assert(xs.numdims() == 1);
    history = std::min(history, static_cast<long>(xs.dims(0)));
    af::array mem = af::constant(0, history, xs.dims(0), xs.type());
    for (long i = 0; i < history; ++i) mem(i, af::seq(i, af::end)) = xs(af::seq(-i - 1));
    return mem;
}

}  // end namespace esn
