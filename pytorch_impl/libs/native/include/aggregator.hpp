/**
 * @file   aggregator.hpp
 * @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
 *
 * @section LICENSE
 *
 * Copyright © 2018-2019 École Polytechnique Fédérale de Lausanne (EPFL).
 * All rights reserved.
 *
 * @section DESCRIPTION
 *
 * Aggregator automated dispatch base class.
**/

#pragma once

// Compiler version check
#if __cplusplus < 201103L
    #error This translation unit requires at least a C++11 compiler
#endif
#ifndef __GNUC__
    #error This translation unit requires a GNU C++ compiler
#endif

// External header
#include <cstddef>
#include <tuple>
#include <utility>
#include <vector>

// Internal header
#include "common.hpp"
#include "pytorch.hpp"

// -------------------------------------------------------------------------- //
// Static aggregation dispatch

/** Static rule base class.
**/
class Rule {
public:
    /** Deleted copy constructor/assignment.
    **/
    Rule(Rule const&) = delete;
    Rule& operator=(Rule const&) = delete;
    /** Deleted default constructor.
    **/
    Rule() = delete;
public:
    /** Always passing assertion.
     * @param ... <Ignored>
    **/
    template<class... Args> static void aggregate_assert(::std::vector<::torch::Tensor> const&, Args&&...) {}
public:
    /** Unsupported aggregations.
     * @param ... <Ignored>
     * @return <No return>
    **/
    template<class... Args> static ::torch::Tensor aggregate_cpu_float(::std::vector<::torch::Tensor> const&, Args&&...) {
        AT_ERROR("Unsupported aggregation of 'float' tensors on CPU with the given additional parameters");
    }
    template<class... Args> static ::torch::Tensor aggregate_cpu_double(::std::vector<::torch::Tensor> const&, Args&&...) {
        AT_ERROR("Unsupported aggregation of 'double' tensors on CPU with the given additional parameters");
    }
    template<class... Args> static ::torch::Tensor aggregate_gpu_float(::std::vector<::torch::Tensor> const&, Args&&...) {
        AT_ERROR("Unsupported aggregation of 'float' tensors on GPU with the given additional parameters");
    }
    template<class... Args> static ::torch::Tensor aggregate_gpu_double(::std::vector<::torch::Tensor> const&, Args&&...) {
        AT_ERROR("Unsupported aggregation of 'double' tensors on GPU with the given additional parameters");
    }
};

/** Static aggregation dispatch class.
 * @param Rule (Static) rule class to use
**/
template<class Rule> class Aggregator final {
public:
    /** Deleted copy constructor/assignment.
    **/
    Aggregator(Aggregator<Rule> const&) = delete;
    Aggregator& operator=(Aggregator<Rule> const&) = delete;
    /** Deleted default constructor.
    **/
    Aggregator() {}
public:
    /** Aggregate the given tensor(s).
     * @param inputs Given tensors(s), at least one, must all be continuous 1-dim tensor(s) of same (non-null) size, data type and device
     * @param ...    Additional, rule-dependent arguments
     * @return Aggregated 1-dim tensor of same size, data type and device
    **/
    template<class... Args> static ::torch::Tensor aggregate(::std::vector<::torch::Tensor> const& inputs, Args&&... args) {
        // Gather dispatch information
        ::caffe2::TypeMeta dtype;
        bool is_cuda = false; // Initially set to avoid spurious warning on read before write
#ifndef NDEBUG
        { // Assertions
            ASSERT(inputs.size() > 0, "Tensor list 'inputs' must contain at least one tensor");
            decltype(inputs[0].size(0)) dim0;
            size_t i = 0;
            for (auto&& input: inputs) {
                ASSERT(input.is_contiguous(), "Tensor 'inputs[", i, "]' must be contiguous");
                ASSERT(input.dim() == 1, "Tensor 'inputs[", i, "]' must be 1D, got ", input.dim(), "D");
                if (i == 0) {
                    dim0 = input.size(0);
                    ASSERT(dim0 > 0, "Tensor 'inputs[", i, "]' must be of positive size, got 0");
                    dtype = input.dtype();
                    is_cuda = input.is_cuda();
                } else {
                    ASSERT(input.size(0) == dim0, "Tensor 'inputs[", i, "]' must be of size ", dim0, ", got ", input.size(0));
                    ASSERT(input.dtype() == dtype, "Tensor 'inputs[", i, "]' must be of type '", dtype.name(), "', got '", input.dtype().name(), "'");
                    ASSERT(input.is_cuda() == is_cuda, "Tensor 'inputs[", i, "]' must reside on ", (is_cuda ? "CPU" : "GPU"), " memory");
                }
                ++i;
            }
        }
#else
        dtype = inputs[0].dtype();
        is_cuda = inputs[0].is_cuda();
#endif
        // Common rule assertions
        Rule::aggregate_assert(inputs, const_cast<Args const&>(args)...);
        // Dispatch call
        if (dtype == ::caffe2::TypeMeta::Make<float>()) {
            if (is_cuda)
                return Rule::aggregate_gpu_float(inputs, ::std::forward<Args>(args)...);
            return Rule::aggregate_cpu_float(inputs, ::std::forward<Args>(args)...);
        }
        if (dtype == ::caffe2::TypeMeta::Make<double>()) {
            if (is_cuda)
                return Rule::aggregate_gpu_double(inputs, ::std::forward<Args>(args)...);
            return Rule::aggregate_cpu_double(inputs, ::std::forward<Args>(args)...);
        }
        AT_ERROR("Unsupported tensor type '", dtype.name(), "'");
    }
};
