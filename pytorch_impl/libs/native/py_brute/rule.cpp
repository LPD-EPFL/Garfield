/**
 * @file   rule.cpp
 * @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
 *
 * @section LICENSE
 *
 * Copyright © 2018-2019 École Polytechnique Fédérale de Lausanne (EPFL).
 * All rights reserved.
 *
 * @section DESCRIPTION
 *
 * Brute GAR, Python binding.
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
**/
void Brute::aggregate_assert(::std::vector<::torch::Tensor> const& inputs, size_t f) {
#ifndef NDEBUG
    auto const n = inputs.size();
    if (unlikely(f == 0))
        AT_ERROR("Brute does not support aggregating gradients when f = 0; use mere averaging in this case");
    if (unlikely(n < 2 * f + 1))
        AT_ERROR("Brute n = ", n, " is too small for f = ", f, "; expected n ≥ ", 2 * f + 1);
#endif
}

// -------------------------------------------------------------------------- //
// Python bind

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("aggregate", Aggregator<Brute>::aggregate<size_t>, "Brute aggregation dispatch");
}
