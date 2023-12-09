// af::array mapped by std::string
#pragma once

#include "arrayfire_utils.hpp"

#include <arrayfire.h>
#include <map>
#include <range/v3/all.hpp>
#include <range/v3/view/enumerate.hpp>
#include <set>

namespace esn {

namespace rg = ranges;
namespace rgv = ranges::views;

class data_map {
private:
    std::set<std::string> keys_;
    af::array data_;

    template <typename Rng>
    static std::map<std::string, double> key_indices(Rng&& keys, size_t offset = 0)
    {
        return rgv::zip(keys, rgv::iota(offset)) | rg::to<std::map<std::string, double>>;
    }

    static af::array make_index_array(const std::map<std::string, double>& indices)
    {
        std::vector<double> ordered_indices = rgv::values(indices) | rg::to_vector;
        return af_utils::to_array(ordered_indices);
    }

public:
    data_map() = default;

    data_map(std::set<std::string> keys, af::array data)
    {
        assert(data.dims(0) == keys.size());
        assert(data.numdims() <= 2);
        keys_ = std::move(keys);
        data_ = std::move(data);
    }

    data_map(const std::map<std::string, af::array>& data_vectors)
    {
        if (data_vectors.empty()) return;
        long data_len = -1;
        for (const auto& [k, v] : data_vectors) {
            assert(v.numdims() == 1);
            assert(data_len == -1 || v.dims(0) == data_len);
            data_len = v.dims(0);
        }
        data_ = af::array(data_vectors.size(), data_len, data_vectors.begin()->second.type());
        for (const auto& [i, v] : rgv::enumerate(rgv::values(data_vectors))) {
            data_(i, af::span) = v.T();
        }
        keys_ = rgv::keys(data_vectors) | rg::to<std::set>;
    }

    data_map(const std::vector<std::string>& keys, const af::array& data)
    {
        assert(data.dims(0) == keys.size());
        assert(data.numdims() <= 2);
        const std::map<std::string, double> ordered_indices = key_indices(keys);
        const af::array index_array = make_index_array(ordered_indices);
        keys_.insert(keys.begin(), keys.end());
        data_ = data(index_array, af::span);
    }

    data_map(const std::string& key, const af::array& data_vector)
    {
        assert(data_vector.numdims() == 1);
        keys_.insert(key);
        data_ = data_vector.T();
    }

    template <typename Selector>
    data_map select(Selector sel) const
    {
        if (keys_.empty()) return {};
        return {keys_, data_(af::span, sel)};
    }

    af::array at(const std::string& key) const
    {
        assert(keys_.contains(key));
        long idx = std::distance(keys_.begin(), keys_.find(key));
        assert(data_(idx).dims(0) == 1);
        return data_(idx, af::span).T();
    }

    template <typename Rng>
    data_map filter(Rng&& keys_rng) const
    {
        std::set<std::string> keys = keys_rng | rg::to<std::set<std::string>>;
        if (keys == keys_) return *this;

        const std::map<std::string, double> indices = key_indices(keys_);
        std::vector<double> filtered_indices;
        for (const std::string& k : keys) {
            assert(indices.contains(k));
            filtered_indices.push_back(indices.at(k));
        }
        const af::array index_array = af_utils::to_array(filtered_indices);
        return {std::move(keys), data_(index_array, af::span)};
    }

    data_map shift(long shift) const
    {
        if (keys_.empty()) return *this;
        af::array shifted = af::shift(data_, 0, shift);
        if (shift < 0)
            shifted(af::span, af::seq(af::end - (-shift), af::end)) = af::NaN;
        else if (shift > 0)
            shifted(af::span, af::seq(0, shift - 1)) = af::NaN;
        return {keys_, std::move(shifted)};
    }

    std::vector<data_map> split(const std::vector<long>& sizes) const
    {
        assert(!keys_.empty());
        std::vector<af::array> data_groups = af_utils::split_data(data_, sizes, 1);
        std::vector<data_map> result;
        for (af::array& group : data_groups) result.emplace_back(keys_, std::move(group));
        return result;
    }

    data_map extend(const data_map& rhs) const
    {
        if (keys_.empty()) return rhs;
        if (rg::includes(keys_, rhs.keys_)) return *this;

        assert(length() == rhs.length());
        af::array joined_data = af::join(0, data_, rhs.data());
        std::map<std::string, double> rhs_indices = key_indices(rhs.keys_, keys_.size());
        std::map<std::string, double> joined_indices = key_indices(keys_);
        joined_indices.insert(rhs_indices.begin(), rhs_indices.end());
        af::array index_array = make_index_array(joined_indices);

        return {rgv::keys(joined_indices) | rg::to<std::set>, joined_data(index_array, af::span)};
    }

    data_map concat(const data_map& rhs) const
    {
        assert(keys_ == rhs.keys_);
        return {keys_, af::join(1, data_, rhs.data_)};
    }

    const std::set<std::string>& keys() const
    {
        return keys_;
    }

    const af::array& data() const
    {
        return data_;
    }

    long size() const
    {
        return keys_.size();
    }

    bool empty() const
    {
        return keys_.empty();
    }

    long length() const
    {
        return data_.dims(1);
    }

    bool contains(const std::string& key) const
    {
        return keys_.contains(key);
    }

    void clear()
    {
        keys_.clear();
        data_ = af::array{};
    }
};

}  // namespace esn