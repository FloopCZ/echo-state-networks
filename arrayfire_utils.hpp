#pragma once

// Arrayfire helper functions. //

#include <arrayfire.h>
#include <cassert>
#include <memory>
#include <string>

namespace af_utils {

/// Return arrayfire info as std::string.
std::string info_string()
{
    std::unique_ptr<const char[]> af_info_ptr{af::infoString()};
    std::string str(af_info_ptr.get());
    return str;
}

/// Return the elementwise square of the given array.
af::array square(const af::array& in)
{
    return in * in;
}

/// Convert af::array of a floating point type to std::vector<double>.
std::vector<double> to_vector(const af::array& af_data)
{
    std::vector<double> host_data(af_data.elements());
    af_data.as(af::dtype::f64).host(host_data.data());
    return host_data;
}

/// Convert std::vector<double> to af::array of type af::dtype::f64.
af::array to_array(const std::vector<double>& data)
{
    assert(!data.empty());
    return af::array(data.size(), data.data());
}

/// Clip the given array to the given interval.
af::array clip(af::array in, double lower, double upper)
{
    in(in < lower) = lower;
    in(in > upper) = upper;
    return in;
}

/// Reduce array to euclidean norm over the given dimension.
af::array norm(const af::array& in, long dim = 0)
{
    return af::sqrt(af::sum(af_utils::square(in), dim));
}

/// Covariance of two arrays
///
/// If the inputs are matrices, it is a pairwise covariance of the corresponding columns.
af::array cov(af::array xs, af::array ys, long dim = 0)
{
    assert(xs.dims() == ys.dims());
    assert(xs.numdims() <= 2);
    assert(dim == 0 || dim == 1);
    // subtract the mean from each row/column
    if (dim == 0) {
        xs = xs - af::tile(af::mean(xs, 0), xs.dims(0), 1);
        ys = ys - af::tile(af::mean(ys, 0), ys.dims(0), 1);
    } else {
        xs = xs - af::tile(af::mean(xs, 1), 1, xs.dims(1));
        ys = ys - af::tile(af::mean(ys, 1), 1, ys.dims(1));
    }
    return af::mean(xs * ys, dim)
      / (af::stdev(xs, AF_VARIANCE_DEFAULT, dim) * af::stdev(ys, AF_VARIANCE_DEFAULT, dim));
}

/// Mean squared error of two arrays.
af::array mse(const af::array& ys_predict, const af::array& ys_truth, long dim = -1)
{
    assert(ys_predict.dims() == ys_truth.dims());
    return af::mean(af_utils::square(ys_predict - ys_truth), dim);
}

template <typename T>
T mse(const af::array& ys_predict, const af::array& ys_truth)
{
    assert(ys_predict.dims() == ys_truth.dims());
    return (mse(af::flat(ys_predict), af::flat(ys_truth)) /* scalar */).scalar<double>();
}

/// Normalized mean squared error of two arrays.
af::array nmse(const af::array& ys_predict, const af::array& ys_truth, long dim = -1)
{
    assert(ys_predict.dims() == ys_truth.dims());
    return af_utils::mse(ys_predict, ys_truth, dim) / af::var(ys_truth, AF_VARIANCE_DEFAULT, dim);
}

template <typename T>
T nmse(const af::array& ys_predict, const af::array& ys_truth)
{
    return (nmse(af::flat(ys_predict), af::flat(ys_truth)) /* scalar */).scalar<double>();
}

/// Normalized root mean squared error of two arrays.
af::array nrmse(const af::array& ys_predict, const af::array& ys_truth, long dim = -1)
{
    assert(ys_predict.dims() == ys_truth.dims());
    return af::sqrt(nmse(ys_predict, ys_truth, dim));
}

template <typename T>
T nrmse(const af::array& ys_predict, const af::array& ys_truth)
{
    return (nrmse(af::flat(ys_predict), af::flat(ys_truth)) /* scalar */).scalar<double>();
}

/// Add a row/column of ones to a matrix.
af::array add_ones(const af::array& A, long dim = 1)
{
    assert(dim == 0 || dim == 1);
    long dim0 = dim == 0 ? 1 : A.dims(0);
    long dim1 = dim == 0 ? A.dims(1) : 1;
    af::array A1 = af::join(dim, A, af::constant(1, dim0, dim1, A.type()));
    return A1;
}

/// Linear regression training.
///
/// Train X, such as [A | 1] * X == B, where [A | 1] is the
/// matrix A with an extra column of ones.
af::array lstsq_train(const af::array& A, const af::array& B)
{
    assert(B.numdims() <= 2);
    assert(A.dims(0) == B.dims(0));
    // add biases (i.e., 1's) to the last column of the coefficient matrix
    af::array A1 = af::join(1, A, af::constant(1, A.dims(0), A.type()));
    af::array X = af::solve(A1, B);
    assert((X.dims() == af::dim4{A.dims(1) + 1, B.dims(1)}));  // + 1 for biases
    return X;
}

/// Linear regression prediction.
///
/// Return the result of [A | 1] * X, where [A | 1] is the
/// matrix A with an extra column of ones.
af::array lstsq_predict(const af::array& A, const af::array& X)
{
    assert(A.numdims() == 2);
    assert(X.numdims() <= 2);
    assert(A.dims(1) + 1 == X.dims(0));  // + 1 for the bias
    // add biases (i.e., 1's) to the last row of the coefficient matrix
    af::array A1 = af::join(1, A, af::constant(1, A.dims(0), A.type()));
    return af::matmul(A1, X);
}

/// Pad the given 2D array with itself (as if it was periodic).
///
/// \param arr The array to be padded.
/// \param border_h The size of the padding on the left and right.
/// \param border_w The size of the padding on the top and down.
af::array periodic(const af::array& A, int border_h, int border_w)
{
    af::array per_arr = af::tile(A, 3, 3);
    return per_arr(
      af::seq(A.dims(0) - border_h, 2 * A.dims(0) + border_h - 1),
      af::seq(A.dims(1) - border_w, 2 * A.dims(1) + border_w - 1));
}

/// Check if all elements of two arrays are almost equal.
bool almost_equal(const af::array& A, const af::array& B, double eps = 1e-8)
{
    return af::allTrue<bool>(af::abs(A - B) < eps);
}

}  // end namespace af_utils
