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

// TODO rename to _targets
inline af::array extract_keys(
  const std::vector<std::string>& keys,
  const std::vector<std::string>& targets,
  const af::array& data)
{
    assert((long)keys.size() == data.dims(0));
    std::vector<double> target_idxs = targets
      | rgv::transform([&](const std::string& t) -> double { return index_of(t, keys); })
      | rg::to_vector;
    return data(af_utils::to_array(target_idxs), af::span);
}

}  // namespace esn