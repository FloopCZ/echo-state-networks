#pragma once

// Arrayfire helper functions. //

#include <arrayfire.h>
#include <cassert>
#include <memory>
#include <optional>
#include <stdexcept>
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

/// Mean absolute error of two arrays.
af::array mae(const af::array& ys_predict, const af::array& ys_truth, long dim = -1)
{
    assert(ys_predict.dims() == ys_truth.dims());
    return af::mean(af::abs(ys_predict - ys_truth), dim);
}

template <typename T>
T mae(const af::array& ys_predict, const af::array& ys_truth)
{
    assert(ys_predict.dims() == ys_truth.dims());
    return (mae(af::flat(ys_predict), af::flat(ys_truth)) /* scalar */)
      .as(af::dtype::f64)
      .scalar<double>();
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
    return (mse(af::flat(ys_predict), af::flat(ys_truth)) /* scalar */)
      .as(af::dtype::f64)
      .scalar<double>();
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
    return (nmse(af::flat(ys_predict), af::flat(ys_truth)) /* scalar */)
      .as(af::dtype::f64)
      .scalar<double>();
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
    return (nrmse(af::flat(ys_predict), af::flat(ys_truth)) /* scalar */)
      .as(af::dtype::f64)
      .scalar<double>();
}

/// Prepend a row/column of the given constant to a matrix.
af::array add_constant_column(const af::array& A, double value, long dim = 1)
{
    assert(dim == 0 || dim == 1);
    long dim0 = dim == 0 ? 1 : A.dims(0);
    long dim1 = dim == 0 ? A.dims(1) : 1;
    af::array A1 = af::join(dim, af::constant(value, dim0, dim1, A.type()), A);
    return A1;
}

/// Prepend a row/column of zeros to a matrix.
af::array add_zeros(const af::array& A, long dim = 1)
{
    return add_constant_column(A, 0., dim);
}

/// Prepend a row/column of ones to a matrix.
af::array add_ones(const af::array& A, long dim = 1)
{
    return add_constant_column(A, 1., dim);
}

/// Linear regression solver
///
/// Assume X = [1 | X0], where [1 | X0] is a matrix X0 with an extra column of ones.
/// Then trains B, such that X * B == Y, optionally regularizing the coefficients (not the
/// intercept).
af::array solve(
  af::array X,
  af::array Y,
  double l2 = 0.,
  const std::optional<af::array>& training_weights = std::nullopt)
{
    assert(Y.numdims() <= 2);
    assert(X.dims(0) == Y.dims(0));
    assert(X.dims(1) > 1);
    if (training_weights.has_value()) {
        assert((training_weights->dims() == af::dim4{X.dims(0)}));
        af::array sqrt_training_weights = af::sqrt(*training_weights);
        X *= af::tile(sqrt_training_weights, 1, X.dims(1));
        Y *= af::tile(sqrt_training_weights, 1, Y.dims(1));
    }
    if (l2 != 0.) {
        af::array reg = std::sqrt(l2) * af::identity(X.dims(1), X.dims(1), X.type());
        reg(0, 0) = 0.;  // do not regularize intercept
        X = af::join(0, X, std::move(reg));
        Y = af::join(0, Y, af::constant(0, {X.dims(1), Y.dims(1)}, Y.type()));
    }
    af::deviceGC();  // Call garbage collector before solve() to avoid OOM.
    return af::solve(std::move(X), std::move(Y));
}

/// Linear regression training.
///
/// Train B, such as [1 | X] * B == Y, where [1 | X] is the
/// matrix X with an extra column of ones.
af::array lstsq_train(
  const af::array& X,
  const af::array& Y,
  double l2 = 0.0,
  const std::optional<af::array>& training_weights = std::nullopt)
{
    assert(Y.numdims() <= 2);
    assert(X.dims(0) == Y.dims(0));
    // add biases (i.e., 1's) as the first column of the coefficient matrix
    af::array B = solve(add_ones(X, 1), Y, l2, training_weights);
    assert((B.dims() == af::dim4{X.dims(1) + 1, Y.dims(1)}));  // + 1 for biases
    return B;
}

/// Linear regression prediction.
///
/// Return the result of [1| A] * X, where [1 | A] is the
/// matrix A with an extra column of ones.
af::array lstsq_predict(const af::array& A, const af::array& X)
{
    assert(A.numdims() == 2);
    assert(X.numdims() <= 2);
    assert(A.dims(1) + 1 == X.dims(0));  // + 1 for the bias
    // add biases (i.e., 1's) as the first column of the coefficient matrix
    return af::matmul(add_ones(A, 1), X);
}

/// Pad the given 2D array with itself (as if it was periodic).
///
/// \param arr The array to be padded.
/// \param border_h The size of the padding on the left and right.
/// \param border_w The size of the padding on the top and down.
af::array periodic(const af::array& A, int border_h, int border_w)
{
    int h = A.dims(0);
    int w = A.dims(1);
    assert(h >= border_h);
    assert(w >= border_w);
    af::array per_arr = af::array(A.dims(0) + border_h * 2, A.dims(1) + border_w * 2, A.type());

    // left side
    per_arr(af::seq(0, border_h - 1), af::seq(0, border_w - 1)) =
      A(af::seq(h - border_h, h - 1), af::seq(w - border_w, w - 1));
    per_arr(af::seq(border_h, border_h + h - 1), af::seq(0, border_w - 1)) =
      A(af::span, af::seq(w - border_w, w - 1));
    per_arr(af::seq(border_h + h, border_h + h + border_h - 1), af::seq(0, border_w - 1)) =
      A(af::seq(0, border_h - 1), af::seq(w - border_w, w - 1));

    // bottom
    per_arr(
      af::seq(border_h + h, border_h + h + border_h - 1), af::seq(border_w, border_w + w - 1)) =
      A(af::seq(0, border_h - 1), af::span);
    per_arr(
      af::seq(border_h + h, border_h + h + border_h - 1),
      af::seq(border_w + w, border_w + w + border_w - 1)) =
      A(af::seq(0, border_h - 1), af::seq(0, border_w - 1));

    // right
    per_arr(
      af::seq(border_h, border_h + h - 1), af::seq(border_w + w, border_w + w + border_w - 1)) =
      A(af::span, af::seq(0, border_w - 1));
    per_arr(af::seq(0, border_h - 1), af::seq(border_w + w, border_w + w + border_w - 1)) =
      A(af::seq(h - border_h, h - 1), af::seq(0, border_w - 1));

    // top
    per_arr(af::seq(0, border_h - 1), af::seq(border_w, border_w + w - 1)) =
      A(af::seq(h - border_h, h - 1), af::span);

    // center
    per_arr(af::seq(border_h, border_h + h - 1), af::seq(border_w, border_w + w - 1)) = A;

    return per_arr;
}

/// Check if all elements of two arrays are almost equal.
bool almost_equal(const af::array& A, const af::array& B, double eps = 1e-8)
{
    return af::allTrue<bool>(af::abs(A - B) < eps);
}

/// Split the data along the given dimension.
std::vector<af::array> split_data(const af::array& data, const std::vector<long>& sizes, long dim)
{
    assert(!sizes.empty());
    std::vector<af::array> groups;
    groups.reserve(sizes.size());
    long begin = 0;
    for (long size : sizes) {
        if (size == 0)
            groups.push_back({});
        else if (dim == 0)
            groups.push_back(data(af::seq(begin, begin + size - 1)));
        else if (dim == 1)
            groups.push_back(data(af::span, af::seq(begin, begin + size - 1)));
        else
            throw std::invalid_argument{"Unsupported dimension in split_data."};
        begin += size;
    }
    return groups;
}

/// Shift array and set the overflown items to the specified value
af::array shift(const af::array& data, long shift, long dim, double fill = af::NaN)
{
    // nan selector for positive and negative shifts
    af::seq negative_selector(af::end - (-shift) + 1, af::end);
    af::seq positive_selector(0, shift - 1);
    // Avoid https://github.com/arrayfire/arrayfire/issues/3532
    assert(data.dims(0) > 1 && "af::shift is broken for dims(0) == 1");
    if (dim == 0) {
        af::array shifted = af::shift(data, shift);
        if (shift < 0)
            shifted(negative_selector) = fill;
        else if (shift > 0)
            shifted(positive_selector) = fill;
        return shifted;
    }
    if (dim == 1) {
        af::array shifted = af::shift(data, 0, shift);
        if (shift < 0)
            shifted(af::span, negative_selector) = fill;
        else if (shift > 0)
            shifted(af::span, positive_selector) = fill;
        return shifted;
    }
    throw std::invalid_argument{"Unsupported dimension in nan_shift."};
}

/// Random shuffle of the rows of a matrix (or of a vector).
af::array shuffle(const af::array& data, af::randomEngine& af_prng)
{
    assert(data.isvector());
    af::array tmp = af::randu(data.dims(0), af::dtype::f32, af_prng);
    af::array val, idx;
    af::sort(val, idx, tmp);
    return data(idx, af::span);
}

}  // end namespace af_utils
