#pragma once

#include "arrayfire.h"

af::array lcnn_step(const af::array& state, const af::array& reservoir_w);