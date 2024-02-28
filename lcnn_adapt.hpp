#pragma once

#include "arrayfire.h"

namespace esn {

struct lcnn_adaptation_config {
    double learning_rate;
    double weight_leakage;
};

af::array lcnn_adapt(
  const af::array& state_memory, const af::array& reservoir_w, const lcnn_adaptation_config& cfg);

}  // namespace esn