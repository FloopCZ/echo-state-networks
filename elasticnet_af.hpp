#pragma once

#include <algorithm>
#include <arrayfire.h>
#include <cassert>
#include <fmt/format.h>
#include <numeric>
#include <random>
#include <stdexcept>

namespace elasticnet_af {

/// Generate a geometric sequence of numbers (based on NumPy geomspace).
///
/// Expects two matrices of dimensions (1, n) with desired start and end values.
/// Returns a matrix of dimensions (num, n) with the corresponding geometric sequence in each
/// column.
inline af::array geomspace(const af::array& start, const af::array& end, long num)
{
    assert(start.dims(0) == 1 && end.dims(0) == 1 && start.elements() == end.elements());
    long n_seqs = start.elements();
    af::array flip_mask = af::flat(end < start);
    af::array log_start = af::log10(af::min(start, end));
    af::array log_end = af::log10(af::max(start, end));
    af::array xs = af::tile(af::seq(num), 1, n_seqs).as(start.type());
    xs = af::pow(10., log_start + xs * (log_end - log_start) / (num - 1));
    xs(af::span, flip_mask) = af::flip(xs(af::span, flip_mask), 0);
    return xs;
}

/// A randomly shuffled sequence [0, n-1].
inline af::array random_seq(long n, af::randomEngine& af_prng)
{
    af::array tmp = af::randu(n, af::dtype::f32, af_prng);
    af::array out, idx;
    af::sort(out, idx, tmp);
    return idx;
}

/// Return 1 for positive values, 0 for zero and -1 for negative values.
inline af::array signum(const af::array& arr)
{
    return -af::sign(arr) + af::sign(-arr);
}

/// Exception thrown when the gradient descent does not converge.
struct convergence_error : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

class ElasticNet {
protected:
    double lambda_;
    double alpha_;
    double tol_;
    long path_len_;
    long max_grad_steps_;

    af::array mean_;
    af::array std_;
    af::array nonzero_std_;
    af::array B_star_;
    af::array intercept_;

    /// Store the mean and std of the input matrix.
    void store_standardization_stats(const af::array& X)
    {
        mean_ = af::mean(X, 0);
        std_ = af::stdev(X, AF_VARIANCE_DEFAULT, 0);
        nonzero_std_ = af::where(af::flat(std_));
    }

    /// Standardize the input matrix to have zero mean and unit variance.
    ///
    /// Uses the standardization coefficients stored in the object.
    /// Constant columns are left unchanged.
    af::array standardize(af::array X) const
    {
        if (X.dims(1) != mean_.dims(1))
            throw std::invalid_argument(fmt::format(
              "The input matrix has a different number of columns ({}) than the stored mean ({}).",
              X.dims(1), mean_.dims(1)));
        X(af::span, nonzero_std_) -= mean_(0, nonzero_std_);
        // X(af::span, nonzero_std_) /= std_(0, nonzero_std_);
        return X;
    }

public:
    /// Create an ElasticNet object with the given parameters.
    ///
    /// \param lambda The strength of the regularization.
    /// \param alpha The mixing parameter between L1 and L2 regularization. The
    ///              larger the alpha, the more L1 regularization is used.
    /// \param tol The tolerance in the coefficients update for the convergence
    ///            of the algorithm. Will be multiplied by squared Frobenius norm of Y.
    /// \param path_len The number of lambda values to use in the pathwise descent.
    /// \param max_grad_steps The maximum number of gradient steps to take.
    /// \throws std::invalid_argument If the parameters are invalid.
    ElasticNet(
      double lambda,
      double alpha,
      double tol = 1e-8,
      long path_len = 100,
      long max_grad_steps = 10000)
      : lambda_(lambda)
      , alpha_(alpha)
      , tol_(tol)
      , path_len_(path_len)
      , max_grad_steps_(max_grad_steps)
    {
        if (lambda_ <= 0) throw std::invalid_argument("The lambda parameter must be positive.");
        if (alpha_ < 0 || alpha_ > 1)
            throw std::invalid_argument("The alpha parameter must be between 0 and 1.");
        if (tol_ <= 0) throw std::invalid_argument("The tolerance parameter must be positive.");
        if (path_len_ < 0)
            throw std::invalid_argument("The path length parameter must be non-negative.");
        // alpha_ = std::max(1e-15, alpha_);  // Avoid division by zero.
    }

    /// Fit the ElasticNet model to the given input-output data.
    ///
    /// \param X The input matrix with dimensions (n_samples, n_features).
    /// \param Y The output matrix with dimensions (n_samples, n_targets).
    /// \throws std::invalid_argument If the input matrices have incompatible dimensions.
    /// \throws convergence_error If the algorithm does not converge.
    void fit(af::array X, af::array Y)
    {
        const long n_samples = X.dims(0);
        const long n_predictors = X.dims(1);
        const long n_targets = Y.dims(1);
        const af::dtype type = X.type();

        // Check the parameters.
        if (Y.dims(0) != n_samples)
            throw std::invalid_argument(fmt::format(
              "The input matrix X has {} rows, but the output matrix Y has {} rows.", n_samples,
              Y.dims(0)));
        if (Y.type() != type) throw std::invalid_argument("X and Y must have the same data type.");

        // Standardize the predictors.
        store_standardization_stats(X);
        if (nonzero_std_.elements() == 0)
            throw std::invalid_argument("All columns of the input matrix are constant.");
        X = standardize(std::move(X));
        X = X(af::span, nonzero_std_);  // Remove constant columns (std == 0).
        const long n_nonconst_predictors = X.dims(1);

        // Subtract the intercept from the targets.
        intercept_ = af::mean(Y, 0);
        Y -= intercept_;

        // Multiply the tolerance using squared Frobenius norm of Y.
        // const af::array tolerance = tol_ * af::sum(Y * Y, 0);

        // Initial guess are zero coefficients.
        // B_star_ = af::constant(0, n_nonconst_predictors, n_targets, type);

        af::array reg =
          std::sqrt(lambda_ * (1. - alpha_)) * af::identity(X.dims(1), X.dims(1), X.type());
        af::array X_reg = af::join(0, X, std::move(reg));
        af::array Y_reg = af::join(0, Y, af::constant(0, {X.dims(1), Y.dims(1)}, Y.type()));
        B_star_ = af::solve(X_reg, Y_reg);

        // B_star_ = af::solve(X, Y);

        // Generate "the path" of lambda values for each target.
        const af::array lambda_path = [&]() {
            if (path_len_ == 0) return af::array{};
            if (path_len_ == 1 || true) return af::constant(lambda_, path_len_, n_targets, type);
            const af::array lambda_max =
              af::max(af::abs(af::matmulTN(X, Y)), 0) / n_samples / alpha_;
            return geomspace(lambda_max, af::constant(lambda_, 1, n_targets, type), path_len_);
        }();

        // Precompute covariance matrices.
        const af::array X_X_covs = af::matmulTN(X, X);
        const af::array Y_X_covs = af::matmulTN(Y, X);

        std::seed_seq seed_seq{n_predictors, n_targets, n_samples, path_len_};
        std::minstd_rand prng{seed_seq};
        std::vector<long> idxs(n_nonconst_predictors);
        std::iota(idxs.begin(), idxs.end(), 0L);

        // Run the coordinate graient descent.
        for (long path_step = 0; path_step < path_len_; ++path_step) {
            const af::array lambda = lambda_path(path_step, af::span);
            // af::print("lambda", lambda, 10);
            long grad_step = 0;
            for (; grad_step < max_grad_steps_; ++grad_step) {
                af::array B = B_star_;
                std::shuffle(idxs.begin(), idxs.end(), prng);
                for (long j : idxs) {
                    // Use the covariance update rule with the precomputed Gram matrices.
                    af::array cov_update = Y_X_covs(af::span, j).T()
                      - af::matmulTN(X_X_covs(af::span, j), B_star_)
                      + X_X_covs(j, j) * B_star_(j, af::span);
                    af::array soft_update =
                      signum(cov_update) * af::max(af::abs(cov_update) - lambda * alpha_, 0.);
                    B_star_(j, af::span) = soft_update / (X_X_covs(j, j) + lambda * (1. - alpha_));
                }

                // Terminating condition.
                af::array B_star_max = af::max(af::abs(B_star_), 0);
                af::array delta_ratio = af::max(af::abs(B_star_ - B), 0) / B_star_max;
                if (af::count<long>(B_star_max) == 0 || af::allTrue<bool>(delta_ratio < tol_))
                    break;
            }
            // if (grad_step == max_grad_steps_)
            //     throw convergence_error{"ElasticNet has not converged."};
        }

        // Adapt the coefficients and intercept to non-standardized predictors.
        // B_star_ /= af::tile(std_(0, nonzero_std_).T(), 1, n_targets);
        intercept_ -= af::matmul(mean_(0, nonzero_std_), B_star_);
        // Extend the coefficients to the full predictor matrix including the constant columns.
        af::array B_star_full = af::constant(0, n_predictors, n_targets, type);
        B_star_full(nonzero_std_, af::span) = B_star_;
        B_star_ = B_star_full;
    }

    /// Predict the output values for the given input matrix.
    ///
    /// \param X The input matrix with dimensions (n_samples, n_features).
    /// \return The predicted output matrix with dimensions (n_samples, n_targets).
    /// \throws std::logic_error If the model has not been fitted yet.
    /// \throws std::invalid_argument If the input matrix has incompatible dimensions.
    af::array predict(const af::array& X) const
    {
        if (B_star_.isempty()) throw std::logic_error("The model has not been fitted yet.");
        if (X.dims(1) != mean_.dims(1))
            throw std::invalid_argument(fmt::format(
              "The input matrix has a different number of columns ({}) than the fitted matrix "
              "({}).",
              X.dims(1), mean_.dims(1)));
        return intercept_ + af::matmul(X, B_star_);
    }

    /// Compute the ElasticNet cost.
    ///
    /// \param X The input matrix with dimensions (n_samples, n_features).
    /// \param Y The output matrix with dimensions (n_samples, n_targets).
    /// \return The cost for each target dimensions (1, n_targets).
    /// \throws std::logic_error Forwarded from \ref predict.
    /// \throws std::invalid_argument If the output matrix has incompatible dimensions or forwarded
    ///         from \ref predict..
    af::array cost(const af::array& X, const af::array& Y) const
    {
        if (Y.dims(1) != intercept_.dims(1))
            throw std::invalid_argument(fmt::format(
              "The output matrix has a different number of columns ({}) than the fitted matrix "
              "({}).",
              Y.dims(1), intercept_.dims(1)));
        af::array sse = af::sum(af::pow(Y - predict(X), 2.), 0) / 2.;
        af::array l2_penalty = lambda_ * (1. - alpha_) * af::sum(B_star_ * B_star_, 0) / 2.;
        af::array l1_penalty = lambda_ * alpha_ * af::sum(af::abs(B_star_), 0);
        return sse + l2_penalty + l1_penalty;
    }

    /// Return the coefficients of the model.
    ///
    /// \param intercept Whether to include the intercept as the zeroth coefficient.
    /// \return The coefficients matrix with dimensions (n_features, n_targets).
    /// \throws std::logic_error If the model has not been fitted yet.
    af::array coefficients(bool intercept = false) const
    {
        if (B_star_.isempty()) throw std::logic_error("The model has not been fitted yet.");
        if (intercept) return af::join(0, intercept_, B_star_);
        return B_star_;
    }

    /// Return the intercept of the model.
    ///
    /// \return The intercept vector with dimensions (1, n_targets).
    /// \throws std::logic_error If the model has not been fitted yet.
    af::array intercept() const
    {
        if (B_star_.isempty()) throw std::logic_error("The model has not been fitted yet.");
        return intercept_;
    }
};

};  // namespace elasticnet_af