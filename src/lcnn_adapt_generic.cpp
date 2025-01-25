#include "lcnn_adapt_generic.hpp"

#include <stdexcept>

namespace esn {

af::array lcnn_adapt_generic(
  const af::array& state_memory, const af::array& reservoir_w, const lcnn_adaptation_config& cfg)
{
    throw std::runtime_error{"LCNN adapt is not implemented for non-CUDA backend."};
}

}  // namespace esn