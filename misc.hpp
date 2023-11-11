#pragma once

#include "arrayfire_utils.hpp"

#include <arrayfire.h>
#include <range/v3/all.hpp>
#include <vector>

namespace esn {

namespace rg = ranges;
namespace rgv = ranges::views;

template <typename T>
long index_of(const T& value, const std::vector<T>& keys)
{
    return std::distance(keys.begin(), std::find(keys.begin(), keys.end(), value));
}

}  // namespace esn