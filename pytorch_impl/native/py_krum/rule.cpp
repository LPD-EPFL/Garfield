/**
 * @file   rule.cpp
 * @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
 *
 * @section LICENSE
 *
 * Copyright © 2018-2019 École Polytechnique Fédérale de Lausanne (EPFL).
 * See LICENSE file.
 *
 * @section DESCRIPTION
 *
 * Multi-Krum GAR, Python binding.
 *
 * Based on the algorithm introduced in the following paper:
 *   Blanchard Peva, El Mhamdi El Mahdi, Guerraoui Rachid, and Stainer Julien.
 *   Machine learning with adversaries: Byzantine tolerant gradient descent.
 *   In Advances in Neural Information Processing Systems 30, pp.118–128.
 *   Curran Associates, Inc., 2017.
**/

// Compiler version check
#if __cplusplus < 201103L
    #error This translation unit requires at least a C++11 compiler
#endif
#ifndef __GNUC__
    #error This translation unit requires a GNU C++ compiler
#endif

// Internal headers
#include <aggregator.hpp>
#include <common.hpp>
#include <pytorch.hpp>
#include "rule.hpp"

// -------------------------------------------------------------------------- //
// Common assertions

/** Assert the validity of the aggregation parameters.
 * @param inputs Given n tensors(s), at least one, must all be continuous 1-dim tensor(s) of same (non-null) size, data type and device
 * @param f      Number of Byzantine tensor(s) to tolerate, must be positive
 * @param m      Number of lowest-scoring tensor(s) to average as output, must be positive and not greater than n - f - 2
**/
void Krum::aggregate_assert(::std::vector<::torch::Tensor> const& inputs, size_t f, size_t m) {
#ifndef NDEBUG
    auto const n = inputs.size();
    if (unlikely(f == 0))
        AT_ERROR("Multi-Krum does not support aggregating gradients when f = 0; use mere averaging in this case");
    if (unlikely(n < 2 * f + 3))
        AT_ERROR("Multi-Krum n = ", n, " is too small for f = ", f, "; min value is n = ", 2 * f + 3);
    if (unlikely(m == 0))
        AT_ERROR("Multi-Krum received non-sense value m = 0");
    if (unlikely(m > n - f - 2))
        AT_ERROR("Multi-Krum m = ", m, " is too large for n = ", n, " and f = ", f, "; max value is m = ", n - f - 2);
#endif
}

// -------------------------------------------------------------------------- //
// Python bind

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("aggregate", Aggregator<Krum>::aggregate<size_t, size_t>, "Multi-Krum aggregation dispatch");
}
