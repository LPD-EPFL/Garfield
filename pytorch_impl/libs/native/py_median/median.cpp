/**
 * @file   median.cpp
 * @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
 *
 * @section LICENSE
 *
 * Copyright © 2018-2019 École Polytechnique Fédérale de Lausanne (EPFL).
 * All rights reserved.
 *
 * @section DESCRIPTION
 *
 * Non-finite-proof median GAR, native C++ implementation.
**/

// Compiler version check
#if __cplusplus < 201103L
    #error This translation unit requires at least a C++11 compiler
#endif
#ifndef __GNUC__
    #error This translation unit requires a GNU C++ compiler
#endif

// External headers
#include <utility>
#include <vector>

// Internal headers
#include <common.hpp>
#include <operations.hpp>
#include <threadpool.hpp>
#include <pytorch.hpp>
#include "rule.hpp"

// -------------------------------------------------------------------------- //
// Native implementation
namespace {

/** Aggregate the given tensor(s).
 * @param inputs Given n tensors(s), at least one, must all be continuous 1-dim tensor(s) of same (non-null) size, data type and device
 * @return Aggregated 1-dim tensor of same size, data type and device
**/
template<class T> ::torch::Tensor aggregate(::std::vector<::torch::Tensor> const& inputs) {
    // Setup
    auto inputs_data = vlarray<T const*>(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i)
        inputs_data[i] = inputs[i].data_ptr<T>();
    auto output = ::torch::empty_like(inputs[0]);
    auto output_data = output.data_ptr<T>();
    // Non-finite-proof median coordinate-by-coordinate
    parallel_for(pool, 0, inputs[0].size(0), [&](size_t start, size_t stop) {
        auto length = inputs.size();
        auto copy = vlarray<T>(length); // Fine since usually not many threads (~10)
        for (size_t x = start; x < stop; ++x) { // Coordinates to work on
            auto coordlength = length;
            // Copy and filter out non-finite values
            auto target = copy.get();
            for (size_t i = 0; i < length; ++i) {
                auto value = inputs_data[i][x];
                if (::std::isfinite(value)) {
                    *(target++) = value;
                } else {
                    --coordlength;
                }
            }
            if (unlikely(coordlength == 0)) { // Only non-finite value for current coordinate
                output_data[x] = 0; // Arbitrary "safe" choice
                continue;
            }
            // Compute the non-finite-proof median
            auto median = copy.get() + coordlength / 2ul;
            ::std::nth_element(copy.get(), median, copy.get() + coordlength, [](T x, T y) { return x < y; });
            output_data[x] = *median;
        }
    });
    // Return
    return output;
}

}
// -------------------------------------------------------------------------- //
// Rule member function definitions

/** Forward to the specialized aggregation function.
 * @param ... Forwarded argument
 * @return Forwarded return value
**/
::torch::Tensor Median::aggregate_cpu_float(::std::vector<::torch::Tensor> const& inputs) {
    return aggregate<float>(inputs);
}
::torch::Tensor Median::aggregate_cpu_double(::std::vector<::torch::Tensor> const& inputs) {
    return aggregate<double>(inputs);
}
