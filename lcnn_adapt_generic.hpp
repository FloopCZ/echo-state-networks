#pragma once

#include "arrayfire.h"
#include "lcnn_adapt.hpp"

namespace esn {

af::array lcnn_adapt_generic(
  const af::array& prev_state,
  const af::array& state,
  const af::array& reservoir_w,
  const lcnn_adaptation_config& adaptation_cfg);

}