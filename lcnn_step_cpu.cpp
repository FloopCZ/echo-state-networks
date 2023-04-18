#include "lcnn_step.hpp"
#include "lcnn_step_generic.hpp"

af::array lcnn_step(const af::array& state, const af::array& reservoir_w)
{
    return lcnn_step_generic(state, reservoir_w);
}