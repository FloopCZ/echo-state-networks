#include "lcnn_adapt.hpp"
#include "lcnn_adapt_generic.hpp"

namespace esn {

af::array lcnn_adapt(
  const af::array& prev_state,
  const af::array& state,
  const af::array& reservoir_w,
  const lcnn_adaptation_config& adaptation_cfg)
{
    return lcnn_adapt_generic(prev_state, state, reservoir_w, adaptation_cfg);
}

}  // namespace esn