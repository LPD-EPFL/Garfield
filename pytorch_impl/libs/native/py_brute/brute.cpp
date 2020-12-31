/**
 * @file   brute.cpp
 * @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
 *
 * @section LICENSE
 *
 * Copyright © 2019 École Polytechnique Fédérale de Lausanne (EPFL).
 * All rights reserved.
 *
 * @section DESCRIPTION
 *
 * Brute GAR, native C++ implementation.
**/

// Compiler version check
#if __cplusplus < 201103L
    #error This translation unit requires at least a C++11 compiler
#endif
#ifndef __GNUC__
    #error This translation unit requires a GNU C++ compiler
#endif

// External headers
#include <cmath>
#include <limits>
#include <mutex>
#include <utility>
#include <vector>

// Internal headers
#include <combinations.hpp>
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
 * @param f      Number of Byzantine tensor(s) to tolerate, must be positive
 * @return Aggregated 1-dim tensor of same size, data type and device
**/
template<class T> ::torch::Tensor aggregate(::std::vector<::torch::Tensor> const& inputs, size_t f) {
    auto output = ::torch::empty_like(inputs[0]);
    auto const n = inputs.size();
    auto distances = vlarray<T>(n * (n - 1) / 2);
    { // Distance computations
        auto cursor = distances.get();
        for (size_t i = 0; i < n - 1; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                auto dist = reduce_sum_squared_difference<T>(inputs[i], inputs[j]);
                if (!::std::isfinite(dist))
                    dist = ::std::numeric_limits<T>::max();
                *(cursor++) = dist;
            }
        }
    }
    size_t global_position;
    { // Smallest-diameter subgroup parallel selection
        auto global_diameter = ::std::numeric_limits<T>::max();
        ::std::mutex lock; // To serialize 'diameter' and 'selected'
        parallel_for(pool, 0, Combinations::count(n, n - f), [&](size_t start, size_t stop) {
            auto local_diameter = ::std::numeric_limits<T>::max();
            size_t local_position;
            // Measure local smallest-diameter subgroup
            Combinations selected{n, n - f};
            auto&& current = selected.get_current();
            for (auto pos = start; pos < stop; ++pos) {
                // Move to position
                if (pos == start) {
                    selected.seek(start);
                } else {
                    selected.next();
                }
                // Measure diameter
                auto diameter = static_cast<T>(0);
                for (size_t i = 0; i < n - f - 1; ++i) {
                    auto const x = current[i];
                    for (size_t j = i + 1; j < n - f; ++j) {
                        auto const y = current[j];
                        // Thanks to 'Combinations', we always have 'x < y' and we can apply the "magic formula" below
                        auto const dist = distances[(2 * n - x - 3) * x / 2 + y - 1];
                        if (dist > diameter)
                            diameter = dist;
                    }
                }
                // Update local if better
                if (diameter < local_diameter) {
                    local_diameter = diameter;
                    local_position = pos;
                }
            }
            { // Replace global smallest-diameter subgroup, if smaller
                ::std::lock_guard<decltype(lock)> guard{lock};
                if (local_diameter < global_diameter) {
                    global_diameter = local_diameter;
                    global_position = local_position;
                }
            }
        });
    }
    { // Average the selected gradients
        Combinations selected{n, n - f};
        selected.seek(global_position);
        selection_average<T>(inputs, output, selected.get_current().data(), n - f);
    }
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
::torch::Tensor Brute::aggregate_cpu_float(::std::vector<::torch::Tensor> const& inputs, size_t f) {
    return aggregate<float>(inputs, f);
}
::torch::Tensor Brute::aggregate_cpu_double(::std::vector<::torch::Tensor> const& inputs, size_t f) {
    return aggregate<double>(inputs, f);
}
