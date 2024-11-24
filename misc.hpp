#pragma once

#include <arrayfire.h>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <range/v3/all.hpp>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace esn {

namespace rg = ranges;
namespace rgv = ranges::views;

inline int days_in_month(const std::tm& date)
{
    std::tm this_month = date;
    this_month.tm_mday = 1;
    time_t this_month_ts = std::mktime(&this_month);

    std::tm next_month = this_month;
    next_month.tm_mon++;
    time_t next_time_ts = std::mktime(&next_month);

    int days_in_month = (next_time_ts - this_month_ts) / (24 * 60 * 60);
    return days_in_month;
}

inline bool is_leap_year(int year)
{
    if (year % 400 == 0) return true;
    if (year % 100 == 0) return false;
    if (year % 4 == 0) return true;
    return false;
}

template <typename T>
long index_of(const T& value, const std::vector<T>& keys)
{
    return std::distance(keys.begin(), std::find(keys.begin(), keys.end(), value));
}

class Timer {
private:
    struct TimeRecord {
        std::chrono::steady_clock::time_point start_time;
        std::chrono::duration<double> duration;
        bool is_running;
    };

    std::unordered_map<std::string, TimeRecord> records_;
    std::ostream* stream_output_;
    std::ofstream csv_file_;

public:
    Timer(std::ostream& output = std::cout, const std::string& csv_path = "")
      : stream_output_(&output)
    {
        if (!csv_path.empty()) {
            csv_file_.open(csv_path, std::ios::out | std::ios::app);
            if (csv_file_.is_open())
                csv_file_ << "Label,Duration (seconds)\n";
            else
                throw std::ios_base::failure("Failed to open CSV file.");
        }
    }
    void start(const std::string& label, bool accumulate = false)
    {
        auto& record = records_[label];
        if (record.is_running)
            throw std::logic_error("Benchmark already started for label: " + label);
        if (!accumulate) record.duration = std::chrono::duration<double>(0);
        records_[label].start_time = std::chrono::steady_clock::now();
        records_[label].is_running = true;
    }

    void output(const std::string& label)
    {
        auto it = records_.find(label);
        if (it == records_.end())
            throw std::logic_error("Benchmark never started for label: " + label);
        if (it->second.is_running)
            throw std::logic_error("Benchmark still running for label: " + label);

        auto& record = it->second;
        *stream_output_ << std::fixed << std::setprecision(6) << "Timer [" << label
                        << "]: " << record.duration.count() << " seconds." << std::endl;
        if (csv_file_.is_open()) csv_file_ << label << "," << record.duration.count() << "\n";
    }

    void stop(const std::string& label, bool quiet = false)
    {
        auto it = records_.find(label);
        if (it == records_.end() || !it->second.is_running)
            throw std::logic_error("Benchmark not started for label: " + label);

        auto& record = it->second;
        record.duration += std::chrono::steady_clock::now() - record.start_time;
        record.is_running = false;

        if (!quiet) output(label);
    }
};

}  // namespace esn