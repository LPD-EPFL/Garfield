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
 * Non-finite-proof median GAR, Python binding.
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
// Python bind

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("aggregate", Aggregator<Median>::aggregate<>, "Non-finite-proof median aggregation dispatch");
}
