#pragma once

#include "arrayfire.h"
#include "lcnn_adapt.hpp"

namespace esn {

af::array lcnn_adapt_generic(
  const af::array& state_memory, const af::array& reservoir_w, const lcnn_adaptation_config& cfg);

}