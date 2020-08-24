/**
 * @file   rule.hpp
 * @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
 *
 * @section LICENSE
 *
 * Copyright © 2018-2019 École Polytechnique Fédérale de Lausanne (EPFL).
 * See LICENSE file.
 *
 * @section DESCRIPTION
 *
 * Multi-Krum GAR, rule declaration.
 *
 * Based on the algorithm introduced in the following paper:
 *   Blanchard Peva, El Mhamdi El Mahdi, Guerraoui Rachid, and Stainer Julien.
 *   Machine learning with adversaries: Byzantine tolerant gradient descent.
 *   In Advances in Neural Information Processing Systems 30, pp.118–128.
 *   Curran Associates, Inc., 2017.
**/

#pragma once

// Compiler version check
#if __cplusplus < 201103L
    #error This translation unit requires at least a C++11 compiler
#endif
#ifndef __GNUC__
    #error This translation unit requires a GNU C++ compiler
#endif

// External headers
#include <vector>

// Internal headers
#include <aggregator.hpp>
#include <common.hpp>
#include <pytorch.hpp>

// -------------------------------------------------------------------------- //
// Aggregation rule

class Krum final: public Rule {
public:
    static void aggregate_assert(::std::vector<::torch::Tensor> const&, size_t, size_t);
public:
    static ::torch::Tensor aggregate_cpu_float(::std::vector<::torch::Tensor> const&, size_t, size_t);
    static ::torch::Tensor aggregate_cpu_double(::std::vector<::torch::Tensor> const&, size_t, size_t);
#ifdef TORCH_CUDA_AVAILABLE
    static ::torch::Tensor aggregate_gpu_float(::std::vector<::torch::Tensor> const&, size_t, size_t);
    static ::torch::Tensor aggregate_gpu_double(::std::vector<::torch::Tensor> const&, size_t, size_t);
#endif
};
