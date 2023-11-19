// af::array mapped by std::string
#pragma once

#include "arrayfire_utils.hpp"

#include <arrayfire.h>
#include <map>
#include <range/v3/all.hpp>
#include <set>

namespace esn {

namespace rg = ranges;
namespace rgv = ranges::views;

using data_map = std::map<std::string, af::array>;

template <typename Rng>
inline data_map make_data_map(Rng&& keys, const af::array& values)
{
    assert(values.numdims() <= 2);
    assert(rg::size(keys) == values.dims(0));

    auto value_slices = rgv::ints((dim_t)0, values.dims(0))
      | rgv::transform([&](long i) { return values(i, af::span).T(); });
    return rgv::zip(keys, value_slices) | rg::to<data_map>;
}

template <typename Selector>
data_map data_map_select(const data_map& data, const Selector& sel)
{
    data_map result;
    for (const auto& [key, value] : data) {
        assert(value.numdims() == 1);
        result.emplace(key, value(sel));
    }
    return result;
}

template <typename Rng>
inline data_map data_map_filter(const data_map& data, Rng&& keys)
{
    data_map result;
    for (const auto& key : keys) {
        assert(data.contains(key));
        result.emplace(key, data.at(key));
    }
    return result;
}

inline af::array data_map_to_array(const data_map& data)
{
    assert(!data.empty());
    const af::array& front = data.begin()->second;
    assert(front.numdims() == 1);
    af::array result(data.size(), front.dims(0), front.type());
    long i = 0;
    for (const auto& [_, value] : data) {
        result(i++, af::span) = value.T();
    }
    return result;
}

inline std::set<std::string> data_map_keys(const data_map& data)
{
    return rgv::keys(data) | rg::to<std::set<std::string>>;
}

inline long data_map_length(const data_map& data)
{
    assert(!data.empty());
    long len = -1;
    for (const auto& [key, value] : data) {
        assert(value.numdims() == 1);
        assert(len == -1 || len == value.dims(0));
        len = value.dims(0);
    }
    return len;
}

inline data_map data_map_shift(const data_map& data, long shift)
{
    data_map result;
    for (const auto& [key, value] : data) {
        assert(value.numdims() == 1);
        af::array shifted = af::shift(value, shift);
        if (shift < 0)
            shifted(af::seq(af::end - (-shift), af::end)) = af::NaN;
        else if (shift > 0)
            shifted(af::seq(0, shift - 1)) = af::NaN;
        result.emplace(key, std::move(shifted));
    }
    return result;
}

inline std::vector<data_map> split_data(const data_map& data, const std::vector<long>& sizes)
{
    std::vector<data_map> result(sizes.size());
    for (const auto& [key, value] : data) {
        long igroup = 0;
        for (af::array& chunk : af_utils::split_data(value, sizes, 0)) {
            result.at(igroup++)[key] = std::move(chunk);
        }
    }
    return result;
}

inline void data_map_print(const data_map& data)
{
    for (const auto& [key, value] : data) {
        af::print(key.c_str(), value);
    }
}

}  // namespace esn