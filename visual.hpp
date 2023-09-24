#pragma once

#include "arrayfire_utils.hpp"
#include "net.hpp"

#include <arrayfire.h>
#include <boost/algorithm/string.hpp>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <range/v3/all.hpp>
#include <thread>

namespace esn {

namespace fs = std::filesystem;
namespace rg = ranges;
namespace rgv = ranges::views;

class visualizer {
private:
    long time_ = 0;
    long sleepms_;
    long history_size_;
    long plot_size_;
    long skip_;

    af::array input_history_;
    af::array output_history_;
    af::array state_history_;
    double state_history_min_;
    double state_history_max_;

    af::Window window_;

    /// If the state is only a vector, reshape it to a rectangle.
    static af::array maybe_square(const af::array& state)
    {
        if (state.numdims() == 1) {
            long short_side = std::lround(std::sqrt(state.elements()));
            while (state.elements() % short_side != 0) --short_side;
            return af::moddims(state, short_side, state.elements() / short_side);
        }
        return state;
    }

    /// Normalizes an array to interval [0, 1] and converts it to logscale.
    /// \param min A lower bound for any element (af::min<double>(a) is fine).
    /// \param max An upper bound for any element (af::max<double>(a) is fine).
    static af::array log_normalize(const af::array& a, double min, double max)
    {
        af::array ret = af::log1p(a - min);
        ret /= std::log1p(max - min);
        return af::clamp(ret, 0., 1.);
    }

    /// Stores the state to the history.
    void update_state_history(const af::array& state)
    {
        // allocate
        if (state_history_.isempty())
            state_history_ = af::constant(af::NaN, state.elements(), history_size_, state.type());
        // update
        if (time_ >= history_size_) state_history_ = af::shift(state_history_, 0, -1);
        long last_index = std::min(history_size_ - 1, time_);
        state_history_(af::span, last_index) = af::flat(state);
        state_history_min_ = af::min<double>(state_history_(af::span, af::seq(0, last_index)));
        state_history_max_ = af::max<double>(state_history_(af::span, af::seq(0, last_index)));
    }

    /// Get plottable state matrix.
    /// \param state The state matrix.
    af::array plottable_state(af::array state) const
    {
        state = log_normalize(state, state_history_min_, state_history_max_);
        state = maybe_square(state);
        return state.as(af::dtype::f32);
    }

    /// Get plottable history matrix.
    af::array plottable_state_history() const
    {
        af::array history = log_normalize(state_history_, state_history_min_, state_history_max_);
        history = af::resize(history, plot_size_, plot_size_, AF_INTERP_LOWER);
        return history.as(af::dtype::f32);
    }

    /// Update a plot data of network inputs or outputs.
    void update_plot_history(af::array& plot_data, const af::array& point)
    {
        assert(point.numdims() == 1);
        // allocate
        if (plot_data.isempty()) {
            plot_data = af::constant(af::NaN, history_size_, 2, point.dims(0), point.type());
            af::array time_range = af::seq(0, history_size_ - 1);
            plot_data(af::span, 0, af::span) = af::tile(time_range, 1, 1, point.dims(0));
        }
        // update
        if (time_ >= history_size_) plot_data = af::shift(plot_data, -1);
        long last_index = std::min(history_size_ - 1, time_);
        plot_data(last_index, 0, af::span) = time_;
        plot_data(last_index, 1, af::span) = point;
    }

    af::array plottable_plot_history(const af::array& plot_data, long i) const
    {
        return plot_data(af::span, af::span, i).as(af::dtype::f32);
    }

    void plot_state(const af::array& state)
    {
        window_(0, 0).image(plottable_state(state), "State");
        update_state_history(state);
        window_(0, 1).image(plottable_state_history(), "State History");
        window_(0, 1).setAxesTitles("", "");
    }

    void plot_input(const af::array& input)
    {
        assert(input.numdims() == 1);
        update_plot_history(input_history_, input);
        window_(1, 0).setAxesTitles("Time", "");
        window_(1, 0).setAxesLimits(
          std::max(0L, time_ - history_size_), std::max(time_, history_size_), -1., 1., true);
        for (long i = 0; i < input.dims(0); ++i) {
            window_(1, 0).plot(plottable_plot_history(input_history_, i), "Desired/Input");
        }
    }

    void plot_output(const af::array& output)
    {
        assert(output.numdims() == 1);
        update_plot_history(output_history_, output);
        window_(1, 1).setAxesTitles("Time", "");
        window_(1, 1).setAxesLimits(
          std::max(0L, time_ - history_size_), std::max(time_, history_size_), -1., 1., true);
        for (long i = 0; i < output.dims(0); ++i) {
            window_(1, 1).plot(plottable_plot_history(output_history_, i), "Feedback/Output");
        }
    }

    void on_state_change(net_base& net, const esn::net_base::on_state_change_data& data)
    {
        if (window_.close()) return;
        if (time_ > skip_) {
            // Print various statistics.
            std::cout << "time: " << time_ << "\n";
            double min = af::min<double>(data.state);
            std::cout << "min activation: " << min << "\n";
            double max = af::max<double>(data.state);
            std::cout << "max activation: " << max << "\n";
            double mean = af::mean<double>(data.state);
            std::cout << "mean activation: " << mean << "\n";
            // Render the image.
            plot_state(data.state);
            plot_input(data.desired.value_or(data.input));
            plot_output(data.output);
            window_.show();
            // Sleep.
            if (sleepms_ > 0) std::this_thread::sleep_for(std::chrono::milliseconds{sleepms_});
        }
        ++time_;
    }

public:
    visualizer(long sleepms, long history_size, long plot_size, long skip)
      : sleepms_{sleepms}
      , history_size_{history_size}
      , plot_size_{plot_size}
      , skip_{skip}
      , window_{2 * (int)plot_size_, 2 * (int)plot_size_, "Echo State Network"}
    {
        window_.grid(2, 2);
    }

    /// Register the visualization callback function.
    template <typename Net>
    void register_callback(Net& net)
    {
        net.add_on_state_change(std::bind_front(&visualizer::on_state_change, this));
    }

    /// Check if the window has been closed.
    void wait_for_close()
    {
        while (!window_.close()) {
            std::this_thread::sleep_for(std::chrono::milliseconds{100});
            window_.show();
        }
    }
};

class file_saver {
private:
    long time_ = 0;
    std::ofstream csv_out_;

    void on_state_change(net_base& net, const esn::net_base::on_state_change_data& data)
    {
        if (time_ == 0) {
            std::vector<std::string> header{"time,"};
            assert(data.input.numdims() == 1);
            assert(data.output.numdims() == 1);
            assert(!data.desired || data.desired->numdims() == 1);
            for (long i = 0; i < data.input.dims(0); ++i)
                header.push_back("input-" + std::to_string(i));
            for (long i = 0; i < data.output.dims(0); ++i)
                header.push_back("output-" + std::to_string(i));
            if (data.desired)
                for (long i = 0; i < data.desired->dims(0); ++i)
                    header.push_back("desired-" + std::to_string(i));
            csv_out_ << boost::join(header, ",") << "\n";
        }
        std::vector<std::string> values{std::to_string(time_)};
        for (long i = 0; i < data.input.dims(0); ++i)
            values.push_back(std::to_string(data.input(i).scalar<double>()));
        for (long i = 0; i < data.output.dims(0); ++i)
            values.push_back(std::to_string(data.output(i).scalar<double>()));
        if (data.desired)
            for (long i = 0; i < data.desired->dims(0); ++i)
                values.push_back(std::to_string((*data.desired)(i).scalar<double>()));
        csv_out_ << boost::join(values, ",") << "\n";
        ++time_;
    }

public:
    file_saver() = default;

    file_saver(const fs::path& csv_out)
    {
        fs::create_directories(csv_out.parent_path());
        csv_out_.exceptions(std::ios_base::failbit | std::ios_base::badbit);
        csv_out_.open(csv_out);
    }

    /// Register the visualization callback function.
    template <typename Net>
    void register_callback(Net& net)
    {
        net.add_on_state_change(std::bind_front(&file_saver::on_state_change, this));
    }
};

}  // end namespace esn
