// af::array mapped by std::string
#pragma once

#include "arrayfire_utils.hpp"

#include <arrayfire.h>
#include <filesystem>
#include <fstream>
#include <map>
#include <nlohmann/json.hpp>
#include <range/v3/all.hpp>
#include <range/v3/view/enumerate.hpp>
#include <set>
#include <stdexcept>

namespace esn {

namespace rg = ranges;
namespace rgv = ranges::views;
namespace fs = std::filesystem;

class data_map {
private:
    std::set<std::string> keys_;
    af::array data_;

    std::map<std::string, double> key_indices(const std::set<std::string>& keys) const
    {
        for ([[maybe_unused]] const std::string& key : keys) assert(keys_.contains(key));
        std::vector<double> indices;
        for (const auto& [i, key] : rgv::enumerate(keys_))
            if (keys.contains(key)) indices.push_back(i);
        return rgv::zip(keys, indices) | rg::to<std::map<std::string, double>>;
    }

    template <rg::range Rng>
    static std::map<std::string, double> iota_indices(Rng&& keys, size_t offset = 0)
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
        if (keys.empty() || data.isempty()) return;
        assert(data.dims(0) == (long)keys.size());
        assert(data.numdims() <= 2);
        keys_ = std::move(keys);
        data_ = std::move(data);
    }

    explicit data_map(const std::map<std::string, af::array>& data_vectors)
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
        assert(data.dims(0) == (long)keys.size());
        assert(data.numdims() <= 2);
        const std::map<std::string, double> ordered_indices = iota_indices(keys);
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

    data_map tail(long len) const
    {
        if (keys_.empty()) return {};
        assert(data_.dims(1) >= len);
        return {keys_, data_(af::span, af::seq(af::end - len, af::end))};
    }

    af::array at(const std::string& key) const
    {
        assert(keys_.contains(key));
        long idx = std::distance(keys_.begin(), keys_.find(key));
        assert(data_(idx).dims(0) == 1);
        return data_(idx, af::span).T();
    }

    template <rg::range Rng>
    data_map filter(Rng&& keys_rng) const
    {
        std::set<std::string> keys = keys_rng | rg::to<std::set<std::string>>;
        if (keys == keys_) return *this;

        const std::map<std::string, double> indices = key_indices(keys_rng | rg::to<std::set>);
        const af::array index_array = af_utils::to_array(rgv::values(indices) | rg::to_vector);
        return {std::move(keys), data_(index_array, af::span)};
    }

    data_map shift(long shift, double fill = af::NaN) const
    {
        if (keys_.empty()) return *this;
        af::array shifted;
        if (data_.dims(0) > 1) {
            shifted = af::shift(data_, 0, shift);
        } else {
            // Workaround for https://github.com/arrayfire/arrayfire/issues/3532
            shifted = af::tile(data_, 2);
            shifted = af::shift(shifted, 0, shift);
            shifted = shifted(0, af::span);
        }
        if (shift < 0)
            shifted(af::span, af::seq(af::end - (-shift) + 1, af::end)) = fill;
        else if (shift > 0)
            shifted(af::span, af::seq(0, shift - 1)) = fill;
        return {keys_, std::move(shifted)};
    }

    std::vector<data_map> split(const std::vector<long>& sizes) const
    {
        if (keys_.empty()) return std::vector<data_map>(sizes.size(), *this);
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
        std::map<std::string, double> rhs_indices = iota_indices(rhs.keys_, keys_.size());
        std::map<std::string, double> joined_indices = iota_indices(keys_);
        joined_indices.insert(rhs_indices.begin(), rhs_indices.end());
        af::array index_array = make_index_array(joined_indices);

        return {rgv::keys(joined_indices) | rg::to<std::set>, joined_data(index_array, af::span)};
    }

    /// Drop all data sequences that are fully NaN.
    data_map drop_nan() const
    {
        if (keys_.empty()) return *this;
        af::array nonnan_indicator_arr = !af::allTrue(af::isNaN(data_), 1);
        if (af::allTrue<bool>(nonnan_indicator_arr)) return *this;
        std::vector<double> nonnan_indicators = af_utils::to_vector(nonnan_indicator_arr);
        assert(nonnan_indicators.size() == keys_.size());
        std::set<std::string> nonnan_keys;
        for (const auto& [is_nonnan, key] : rgv::zip(nonnan_indicators, keys_))
            if (is_nonnan) nonnan_keys.insert(key);
        return {std::move(nonnan_keys), data_(nonnan_indicator_arr, af::span)};
    }

    /// Only keep every n-th, others set to NaN.
    data_map every_nth(const std::set<std::string>& keys, long n) const
    {
        assert(n > 0);
        if (keys_.empty()) return *this;
        if (n <= 1) return *this;
        const std::map<std::string, double> indices = key_indices(keys);
        af::array indices_arr = af_utils::to_array(rgv::values(indices) | rg::to_vector);
        // Cast the af::seq to af::array to avoid
        // https://github.com/arrayfire/arrayfire/issues/3525.
        af::array selector = af::seq(0, data_.dims(1) - 1, n);
        af::array data = data_;
        data(indices_arr, af::span) = af::NaN;
        data(indices_arr, selector) = data_(indices_arr, selector);
        return {keys_, std::move(data)};
    }

    /// Probabilistically change values to NaN.
    data_map probably_nan(
      const std::set<std::string>& keys, double p, long keep_tail, af::randomEngine& af_prng) const
    {
        if (keys_.empty()) return *this;
        if (p <= 0.) return *this;
        const std::map<std::string, double> indices = key_indices(keys);
        af::array indices_arr = af_utils::to_array(rgv::values(indices) | rg::to_vector);
        af::array selector = af::constant(false, data_.dims(), af::dtype::b8);
        selector(indices_arr, af::seq(0, af::end - keep_tail)) =
          af::randu({(dim_t)keys.size(), data_.dims(1) - keep_tail}, af::dtype::f32, af_prng) < p;
        af::array data = data_;
        data(selector) = af::NaN;
        return {keys_, std::move(data)};
    }

    data_map normalize_by(const data_map& ref) const
    {
        assert(keys_ == ref.keys_);
        assert(ref.length() > 0);
        if (keys_.empty()) return *this;
        af::array data = data_;
        data -= af::tile(af::mean(ref.data_, 1), 1, length());
        data /= af::tile(af::stdev(ref.data_, AF_VARIANCE_POPULATION, 1), 1, length());
        return {keys_, std::move(data)};
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

    void save(const fs::path& dir)
    {
        fs::create_directories(dir);
        nlohmann::json keys = keys_;
        std::ofstream{dir / "keys.json"} << keys.dump();
        std::string data_file = dir / "data.bin";
        if (!data_.isempty()) af::saveArray("data", data_, data_file.c_str());
    }

    void load(const fs::path& dir)
    {
        if (!fs::exists(dir))
            throw std::runtime_error{"Data map dir " + dir.string() + " does not exist."};
        std::ifstream fin{dir / "keys.json"};
        keys_ = nlohmann::json::parse(fin);
        std::string data_file = dir / "data.bin";
        data_ = af::array{};
        if (fs::exists(data_file)) data_ = af::readArray(data_file.c_str(), "data");
    }
};

}  // namespace esn