#pragma once

#include "arrayfire.h"

namespace esn {

struct lcnn_adaptation_config {
    double learning_rate;
    double weight_leakage;
};

af::array lcnn_adapt(
  const af::array& prev_state,
  const af::array& state,
  const af::array& reservoir_w,
  const lcnn_adaptation_config& adaptation_cfg);

}  // namespace esn