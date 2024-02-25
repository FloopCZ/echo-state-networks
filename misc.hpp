#pragma once

#include "arrayfire_utils.hpp"

#include <arrayfire.h>
#include <range/v3/all.hpp>
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

    int days_in_month = (next_time_ts - this_month_ts) / (24 * 24 * 60);
    return days_in_month;
}

template <typename T>
long index_of(const T& value, const std::vector<T>& keys)
{
    return std::distance(keys.begin(), std::find(keys.begin(), keys.end(), value));
}

}  // namespace esn