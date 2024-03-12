#pragma once

// Ensemble wrapper for lcnn. //

#include "lcnn.hpp"

namespace esn {

template <af::dtype DType = DEFAULT_AF_DTYPE>
class lcnn_ensemble : public net_base {
protected:
    std::vector<std::unique_ptr<lcnn<DType>>> nets_;

    data_map last_output_;

    void update_last_output()
    {
        data_map last_outputs = nets_.at(0)->last_output();
        if (last_outputs.empty()) {
            last_output_.clear();
            return;
        }
        for (long i = 1; i < nets_.size(); ++i) {
            last_outputs = last_outputs.concat(nets_.at(i)->last_output());
        }
        last_output_ = {last_outputs.keys(), af::median(last_outputs.data(), 1)};
        assert(last_output_.keys() == nets_.at(0)->output_names());
        assert(last_output_.length() == 1);
        for (std::unique_ptr<lcnn<DType>>& net : nets_) net->last_output(last_output_);
    }

public:
    lcnn_ensemble(std::vector<std::unique_ptr<net_base>> nets)
    {
        for (std::unique_ptr<net_base>& net : nets)
            nets_.emplace_back(dynamic_cast<lcnn<DType>*>(net.release()));
    }

    void step(
      const data_map& step_input,
      const data_map& step_feedback,
      const data_map& step_desired,
      const data_map& step_meta,
      input_transform_fn_t input_transform) override
    {
        for (std::unique_ptr<lcnn<DType>>& net : nets_)
            net->step(step_input, step_feedback, step_desired, step_meta, input_transform);
        update_last_output();

        // Call the registered callback functions.
        for (on_state_change_callback_t& fnc : on_state_change_callbacks_) {
            on_state_change_data data = {
              .state = nets_.at(0)->state(),
              .input =
                {.input = step_input,
                 .feedback = step_feedback,
                 .desired = step_desired,
                 .meta = step_meta},
              .output = last_output_,
              .event = event_};
            fnc(*this, std::move(data));
        }
        event_ = std::nullopt;
    }

    feed_result_t feed(const input_t& input) override
    {
        long data_len = -1;
        auto check_data = [&data_len](const data_map& dm) {
            if (dm.empty()) return;
            assert(dm.data().type() == DType);
            assert(dm.data().numdims() <= 2);
            assert(dm.size() > 0);
            assert(data_len == -1 || dm.length() == data_len);
            assert(!af::anyTrue<bool>(af::isInf(dm.data())));
            data_len = dm.length();
        };
        check_data(input.input);
        check_data(input.feedback);
        check_data(input.desired);

        feed_result_t result;
        result.states = af::constant(
          af::NaN, nets_.at(0)->state().dims(0), nets_.at(0)->state().dims(1), data_len, DType);
        result.outputs = af::constant(af::NaN, nets_.at(0)->output_names().size(), data_len, DType);
        result.desired = input.desired.data();
        for (long i = 0; i < data_len; ++i) {
            // prepare the inputs for this step
            data_map step_input = input.input.select(i);
            data_map step_feedback = input.feedback.select(i);
            data_map step_desired = input.desired.select(i);
            data_map step_meta = input.meta.select(i);
            step(step_input, step_feedback, step_desired, step_meta, input.input_transform);
            if (!last_output_.empty()) result.outputs(af::span, i) = last_output_.data();
        }
        return result;
    }

    train_result_t train(const input_t& input) override
    {
        for (std::unique_ptr<lcnn<DType>>& net : nets_) net->train(net->feed(input), input);
        return {};
    }

    void reset() override
    {
        for (std::unique_ptr<lcnn<DType>>& net : nets_) net->reset();
    }

    train_result_t train(const feed_result_t data, const input_t& input) override
    {
        for (std::unique_ptr<lcnn<DType>>& net : nets_) net->train(data, input);
        return {};
    }

    void clear_feedback() override
    {
        for (std::unique_ptr<lcnn<DType>>& net : nets_) net->clear_feedback();
    }

    const af::array& state() const override
    {
        return nets_.at(0)->state();
    }

    /// Set the current state of the network.
    void state(af::array new_state) override
    {
        throw std::runtime_error{"Not implemented."};
    }

    const std::set<std::string>& input_names() const override
    {
        return nets_.at(0)->input_names();
    }

    const std::set<std::string>& output_names() const override
    {
        return nets_.at(0)->output_names();
    }

    double neuron_ins() const override
    {
        return nets_.at(0)->neuron_ins();
    }

    void random_noise(bool enable) override
    {
        for (std::unique_ptr<lcnn<DType>>& net : nets_) net->random_noise(enable);
    }

    void learning(bool enable) override
    {
        for (std::unique_ptr<lcnn<DType>>& net : nets_) net->learning(enable);
    }

    std::unique_ptr<net_base> clone() const override
    {
        std::vector<std::unique_ptr<net_base>> net_clones;
        for (const std::unique_ptr<lcnn<DType>>& net : nets_) net_clones.push_back(net->clone());
        return std::make_unique<lcnn_ensemble<DType>>(std::move(net_clones));
    }
};

template <af::dtype DType = DEFAULT_AF_DTYPE>
lcnn_ensemble<DType> random_lcnn_ensemble(
  const std::set<std::string>& input_names,
  const std::set<std::string>& output_names,
  const po::variables_map& args,
  prng_t& prng)
{
    std::vector<std::unique_ptr<net_base>> nets;
    for (long i = 0; i < args.at("lcnn-ensemble.n").as<long>(); ++i) {
        nets.push_back(
          std::make_unique<lcnn<DType>>(random_lcnn(input_names, output_names, args, prng)));
        std::seed_seq sseq{prng(), prng(), prng(), prng()};
        prng.seed(sseq);
    }
    return lcnn_ensemble<DType>{std::move(nets)};
}

inline po::options_description lcnn_ensemble_arg_description()
{
    po::options_description lcnn_arg_desc = lcnn_arg_description();
    lcnn_arg_desc.add_options()  //
      ("lcnn-ensemble.n",
       po::value<long>()->default_value(10),       //
       "The number of networks in the ensemble.")  //
      ;
    return lcnn_arg_desc;
}

inline std::unique_ptr<net_base> make_net(
  const std::set<std::string>& input_names,
  const std::set<std::string>& output_names,
  const po::variables_map& args,
  prng_t& prng)
{
    if (args.at("gen.net-type").as<std::string>() == "lcnn") {
        return std::make_unique<lcnn<>>(random_lcnn(input_names, output_names, args, prng));
    }
    if (args.at("gen.net-type").as<std::string>() == "lcnn-ensemble") {
        return std::make_unique<lcnn_ensemble<>>(
          random_lcnn_ensemble(input_names, output_names, args, prng));
    }
    if (args.at("gen.net-type").as<std::string>() == "simple-esn") {
        return std::make_unique<simple_esn<>>(random_esn(input_names, output_names, args, prng));
    }
    throw std::runtime_error{
      "Unknown net type \"" + args.at("gen.net-type").as<std::string>() + "\"."};
}

}  // namespace esn
