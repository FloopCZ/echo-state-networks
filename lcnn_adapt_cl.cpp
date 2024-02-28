#include "lcnn_adapt.hpp"
#include "lcnn_adapt_generic.hpp"

namespace esn {

af::array lcnn_adapt(
  const af::array& state_memory, const af::array& reservoir_w, const lcnn_adaptation_config& cfg)
{
    return lcnn_adapt_generic(state_memory, reservoir_w, cfg);
}

}  // namespace esn