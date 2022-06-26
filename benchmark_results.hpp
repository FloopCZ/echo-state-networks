#pragma once

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/max.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/variance.hpp>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <range/v3/all.hpp>
#include <string>
#include <vector>

namespace esn {

namespace bacc = boost::accumulators;
namespace rg = ranges;
namespace rgv = ranges::views;

/// A storage class for the result of a single benchmark.
class stats {
public:
    using Stats = bacc::stats<bacc::tag::mean, bacc::tag::max, bacc::tag::variance(bacc::lazy)>;
    using Accumulator = bacc::accumulator_set<double, Stats>;

private:
    std::string name_;
    std::vector<double> values_;
    Accumulator acc_;

    static double nan_to_inf(double value)
    {
        if (std::isnan(value)) return std::numeric_limits<double>::infinity();
        return value;
    }

    static std::vector<double> nan_to_inf(std::vector<double> values)
    {
        for (double& v : values) v = nan_to_inf(v);
        return values;
    }

public:
    stats() = default;

    stats(std::string name, const std::vector<double>& values = {}) : name_{std::move(name)}
    {
        insert(values);
    }

    stats(const std::vector<double>& values)
    {
        insert(values);
    }

    void insert(std::vector<double> values)
    {
        values = nan_to_inf(values);
        for (double v : values) acc_(v);
        values_.insert(values_.end(), values.begin(), values.end());
    }

    void insert(double value)
    {
        return insert(std::vector<double>{value});
    }

    const std::string& name() const
    {
        return name_;
    }

    double mean() const
    {
        return bacc::mean(acc_);
    }

    double std() const
    {
        return std::sqrt(bacc::variance(acc_));
    }

    double median() const
    {
        if (values_.empty()) throw std::runtime_error{"Values cannot be empty for median."};
        std::vector<double> values = nan_to_inf(values_);
        auto middle = values.begin() + (values.size() - 1) / 2;
        std::nth_element(values.begin(), middle, values.end());
        return *middle;
    }

    double max() const
    {
        return bacc::max(acc_);
    }

    double back() const
    {
        return values_.back();
    }
};

/// A storage class for multiple benchmark results.
/// This storage is thread-safe for insertion.
class benchmark_results {
private:
    std::vector<std::string> data_order_;
    std::map<std::string, stats> data_;
    std::mutex mtx_;

public:
    benchmark_results() = default;
    benchmark_results(const benchmark_results& rhs) : data_order_{rhs.data_order_}, data_{rhs.data_}
    {
    }
    benchmark_results(benchmark_results&& rhs)
      : data_order_{std::move(rhs.data_order_)}, data_{std::move(rhs.data_)}
    {
    }

    /// Insert a new result into the storage.
    void insert(const std::string& name, const std::vector<double>& values)
    {
        std::scoped_lock lock{mtx_};
        if (!data_.contains(name)) data_order_.push_back(name);
        data_.emplace(name, name);
        data_.at(name).insert(values);
    }

    /// Insert a new result into the storage.
    void insert(const std::string& name, double value)
    {
        return insert(name, std::vector<double>{value});
    }

    /// Insert a new result into the storage.
    const stats& at(const std::string& name) const
    {
        return data_.at(name);
    }

    /// Get view of the current data.
    auto view() const
    {
        return data_order_ | rgv::transform([this](const std::string& name) -> const stats& {
                   return data_.at(name);
               });
    }

    /// Generate a csv row out of the stored results.
    std::vector<std::string> csv_values() const
    {
        std::vector<std::string> values;
        for (auto& s : view()) {
            values.push_back(std::to_string(s.mean()));
            values.push_back(std::to_string(s.std()));
        }
        return values;
    }

    /// Generate a csv header for the stored results.
    std::vector<std::string> csv_header() const
    {
        std::vector<std::string> values;
        for (auto& s : view()) {
            values.push_back(s.name());
            values.push_back(s.name() + " (std)");
        }
        return values;
    }
};

/// Pretty printing of benchmark results to std::ostream.
std::ostream& operator<<(std::ostream& out, const benchmark_results& results)
{
    for (auto& s : results.view()) {
        out << std::setw(30) << s.name()                      //
            << std::setw(15) << s.mean() << " (+- "           //
            << std::setw(15) << s.std() << ")" << std::endl;  //
    }
    return out;
}

}  // end namespace esn