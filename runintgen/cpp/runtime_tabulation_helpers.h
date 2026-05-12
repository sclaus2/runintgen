#ifndef RUNINTGEN_RUNTIME_TABULATION_HELPERS_H
#define RUNINTGEN_RUNTIME_TABULATION_HELPERS_H

#include "runintgen_runtime_abi.h"

#include <basix/indexing.h>

#include <array>
#include <cstddef>

namespace runintgen::detail
{
inline std::size_t table_shape_size(const std::array<std::size_t, 4>& shape)
{
  return shape[0] * shape[1] * shape[2] * shape[3];
}

inline int derivative_index(const int counts[4], int tdim)
{
  switch (tdim)
  {
  case 1:
    return basix::indexing::idx(counts[0]);
  case 2:
    return basix::indexing::idx(counts[0], counts[1]);
  case 3:
    return basix::indexing::idx(counts[0], counts[1], counts[2]);
  default:
    return -1;
  }
}

inline std::size_t source_dof(const runintgen_table_request& request, int dof)
{
  if (request.offset >= 0 && request.block_size > 0)
    return static_cast<std::size_t>(request.offset + request.block_size * dof);
  if (request.offset >= 0)
    return static_cast<std::size_t>(request.offset + dof);
  return static_cast<std::size_t>(dof);
}
} // namespace runintgen::detail

#endif // RUNINTGEN_RUNTIME_TABULATION_HELPERS_H
