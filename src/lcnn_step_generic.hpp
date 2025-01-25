#pragma once

#include "arrayfire.h"

af::array lcnn_step_generic(const af::array& state, const af::array& reservoir_w);