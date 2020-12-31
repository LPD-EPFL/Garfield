/**
 * @file   rule.hpp
 * @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
 *
 * @section LICENSE
 *
 * Copyright © 2018-2019 École Polytechnique Fédérale de Lausanne (EPFL).
 * All rights reserved.
 *
 * @section DESCRIPTION
 *
 * Bulyan over Multi-Krum GAR, rule declaration.
 *
 * Based on the algorithm introduced in the following paper:
 *   El Mhamdi El Mahdi, Guerraoui Rachid, and Rouault Sébastien.
 *   The Hidden Vulnerability of Distributed Learning in Byzantium.
 *   In Dy, J. and Krause, A. (eds.), Proceedings of the 35th International
 *   Conference on Machine Learning, volume 80 of Proceedings of Machine
 *   Learning  Research, pp. 3521-3530, Stockholmsmässan, Stockholm Sweden,
 *   10-15 Jul 2018. PMLR. URL http://proceedings.mlr.press/v80/mhamdi18a.html.
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

class Bulyan final: public Rule {
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
