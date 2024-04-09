#pragma once

// Fixer network wrapper for lcnn. //

#include "data_map.hpp"
#include "lcnn_ensemble.hpp"
#include "net.hpp"
#include "simple_esn.hpp"

#include <memory>

namespace esn {

template <af::dtype DType = DEFAULT_AF_DTYPE>
class lcnn_fixer : public net_base {
protected:
    std::unique_ptr<lcnn<DType>> predict_net_;
    std::unique_ptr<lcnn<DType>> fixer_net_;

public:
    lcnn_fixer(std::unique_ptr<net_base> predict_net, std::unique_ptr<net_base> fixer_net)
      : predict_net_{dynamic_cast<lcnn<DType>*>(predict_net.release())}
      , fixer_net_{dynamic_cast<lcnn<DType>*>(fixer_net.release())}
    {
    }

    void step(
      const data_map& step_input,
      const data_map& step_feedback,
      const data_map& step_desired,
      const data_map& step_meta,
      input_transform_fn_t input_transform) override
    {
        predict_net_->step(step_input, step_feedback, step_desired, step_meta, input_transform);
        data_map fixer_in = predict_net_->last_output().extend(step_input);
        fixer_net_->step(fixer_in, step_feedback, step_desired, step_meta, input_transform);

        // Call the registered callback functions.
        for (on_state_change_callback_t& fnc : on_state_change_callbacks_) {
            on_state_change_data data = {
              .state = fixer_net_->state(),
              .input =
                {.input = step_input,
                 .feedback = step_feedback,
                 .desired = step_desired,
                 .meta = step_meta,
                 .input_transform = input_transform},
              .output = fixer_net_->last_output(),
              .event = event_};
            fnc(*this, std::move(data));
        }
        event_ = std::nullopt;
    }

    feed_result_t feed(const input_t& input) override
    {
        /*
        feed_result_t predict_result = predict_net_->feed(input);
        data_map predict_output{predict_net_->output_names(), predict_result.outputs};
        data_map fixer_input = predict_output.extend(input.input);
        predict_output = {};
        predict_result = {};
        // Use the predict data to train the fixer.
        return fixer_net_->feed(
          {.input = std::move(fixer_input),
           .feedback = input.feedback,
           .desired = input.desired,
           .meta = input.meta,
           .input_transform = input.input_transform});
        */

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
          af::NaN, predict_net_->state().dims(0), predict_net_->state().dims(1), data_len, DType);
        result.outputs =
          af::constant(af::NaN, predict_net_->output_names().size(), data_len, DType);
        result.desired = input.desired.data();
        for (long i = 0; i < data_len; ++i) {
            // prepare the inputs for this step
            data_map step_input = input.input.select(i);
            data_map step_feedback = input.feedback.select(i);
            data_map step_desired = input.desired.select(i);
            data_map step_meta = input.meta.select(i);
            step(step_input, step_feedback, step_desired, step_meta, input.input_transform);
            if (!fixer_net_->last_output().empty())
                result.outputs(af::span, i) = fixer_net_->last_output().data();
        }
        return result;
    }

    train_result_t train(const input_t& input) override
    {
        // Train the predict net, but keep the original.
        {
            std::unique_ptr<lcnn<DType>> predict_clone{
              dynamic_cast<lcnn<DType>*>(predict_net_->clone().release())};
            predict_clone->train(input);
            // Set the output weights of the original and feed again.
            predict_net_->output_w(predict_clone->output_w());
        }
        feed_result_t predict_result = predict_net_->feed(
          {.input = input.input,
           .feedback = {},
           .desired = input.desired,
           .meta = input.meta,
           .input_transform = input.input_transform});
        data_map predict_output{predict_net_->output_names(), predict_result.outputs};
        data_map fixer_input = predict_output.extend(input.input);
        predict_output = {};
        predict_result = {};
        // Use the predict data to train the fixer.
        return fixer_net_->train(
          {.input = std::move(fixer_input),
           .feedback = input.desired,
           .desired = input.desired,
           .meta = input.meta,
           .input_transform = input.input_transform});
    }

    void reset() override
    {
        predict_net_.reset();
        fixer_net_.reset();
    }

    train_result_t train(const feed_result_t data, const input_t& input) override
    {
        throw std::runtime_error{"Not implemented."};
    }

    void clear_feedback() override
    {
        predict_net_->clear_feedback();
        fixer_net_->clear_feedback();
    }

    const af::array& state() const override
    {
        return fixer_net_->state();
    }

    /// Set the current state of the network.
    void state(af::array new_state) override
    {
        throw std::runtime_error{"Not implemented."};
    }

    const std::set<std::string>& input_names() const override
    {
        return predict_net_->input_names();
    }

    const std::set<std::string>& output_names() const override
    {
        return fixer_net_->output_names();
    }

    double neuron_ins() const override
    {
        return predict_net_->neuron_ins();
    }

    void random_noise(bool enable) override
    {
        predict_net_->random_noise(enable);
        fixer_net_->random_noise(enable);
    }

    void learning(bool enable) override
    {
        predict_net_->learning(enable);
        fixer_net_->learning(enable);
    }

    std::unique_ptr<net_base> clone() const override
    {
        return std::make_unique<lcnn_fixer<DType>>(predict_net_->clone(), fixer_net_->clone());
    }

    void save(const fs::path& dir) override
    {
        predict_net_->save(dir / "predict");
        fixer_net_->save(dir / "fixer");
    }

    static lcnn_fixer<DType> load(const fs::path& dir)
    {
        if (!fs::exists(dir))
            throw std::runtime_error{
              "LCNN fixer snapshot dir `" + dir.string() + "` does not exist."};
        auto predict_net = std::make_unique<lcnn<DType>>(lcnn<DType>::load(dir / "predict"));
        auto fixer_net = std::make_unique<lcnn<DType>>(lcnn<DType>::load(dir / "fixer"));
        return lcnn_fixer<DType>{std::move(predict_net), std::move(fixer_net)};
    }
};

template <af::dtype DType = DEFAULT_AF_DTYPE>
lcnn_fixer<DType> random_lcnn_fixer(
  const std::set<std::string>& input_names,
  const std::set<std::string>& output_names,
  const po::variables_map& args,
  prng_t& prng)
{
    std::unique_ptr<lcnn<DType>> predict_net =
      std::make_unique<lcnn<DType>>(random_lcnn(input_names, output_names, args, prng));
    std::unique_ptr<lcnn<DType>> fixer_net{
      dynamic_cast<lcnn<DType>*>(predict_net->clone().release())};
    return lcnn_fixer<DType>{std::move(predict_net), std::move(fixer_net)};
}

inline po::options_description lcnn_fixer_arg_description()
{
    po::options_description lcnn_arg_desc = lcnn_arg_description();
    lcnn_arg_desc.add_options()                            //
      ("lcnn-fixer.load", po::value<std::string>(),        //
       "Directory with predict and fixer net snapshots.")  //
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
        if (args.contains("lcnn.load")) {
            fs::path net_dir = args.at("lcnn.load").as<std::string>();
            return std::make_unique<lcnn<>>(lcnn<>::load(net_dir));
        }
        return std::make_unique<lcnn<>>(random_lcnn(input_names, output_names, args, prng));
    }
    if (args.at("gen.net-type").as<std::string>() == "lcnn-ensemble") {
        return std::make_unique<lcnn_ensemble<>>(
          random_lcnn_ensemble(input_names, output_names, args, prng));
    }
    if (args.at("gen.net-type").as<std::string>() == "lcnn-fixer") {
        if (args.contains("lcnn-fixer.load")) {
            fs::path net_dir = args.at("lcnn-fixer.load").as<std::string>();
            return std::make_unique<lcnn_fixer<>>(lcnn_fixer<>::load(net_dir));
        }
        return std::make_unique<lcnn_fixer<>>(
          random_lcnn_fixer(input_names, output_names, args, prng));
    }
    if (args.at("gen.net-type").as<std::string>() == "simple-esn") {
        return std::make_unique<simple_esn<>>(random_esn(input_names, output_names, args, prng));
    }
    throw std::runtime_error{
      "Unknown net type \"" + args.at("gen.net-type").as<std::string>() + "\"."};
}

}  // namespace esn
