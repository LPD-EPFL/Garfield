/**
 * @file   krum.cpp
 * @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
 *
 * @section LICENSE
 *
 * Copyright © 2018-2019 École Polytechnique Fédérale de Lausanne (EPFL).
 * All rights reserved.
 *
 * @section DESCRIPTION
 *
 * Multi-Krum GAR, native C++ implementation.
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
 * @param f      Number of Byzantine tensor(s) to tolerate, must be positive
 * @param m      Number of lowest-scoring tensor(s) to average as output, must be positive and not greater than n - f - 2
 * @return Aggregated 1-dim tensor of same size, data type and device
**/
template<class T> ::torch::Tensor aggregate(::std::vector<::torch::Tensor> const& inputs, size_t f, size_t m) {
    // Initialize
    auto const n  = inputs.size();
    auto const kn = n * (n - 1);
    auto const ln = n * (n - 1) / 2;
    auto output = ::torch::empty_like(inputs[0]);
    // Process
    auto pos_to_gradid = vlarray<size_t>(kn);
    auto distances = vlarray<T>(ln);
    { // Distance computations
        auto cursor = distances.get();
        auto poscur = pos_to_gradid.get();
        for (size_t i = 0; i < n - 1; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                *(cursor++) = reduce_sum_squared_difference<T>(inputs[i], inputs[j]);
                *(poscur++) = i;
                *(poscur++) = j;
            }
        }
    }
    auto scores = vlarray<T>(n);
    { // Score computations
        auto ranks = vlarray<size_t>(ln); // Indexes for 'distances', so that 'distances[ranks[i]]' increases with 'i' ('nan' is treated as '+inf')
        { // Compute 'ranks'
            for (size_t i = 0; i < ln; ++i)
                ranks[i] = i;
            ::std::sort(ranks.get(), ranks.get() + ln, [&](size_t a, size_t b) {
                auto&& x = distances[a];
                if (unlikely(!::std::isfinite(x)))
                    return false;
                auto&& y = distances[b];
                if (unlikely(!::std::isfinite(y)))
                    return true;
                return x < y;
            });
        }
        for (size_t i = 0; i < n; ++i) { // Compute 'scores'
            T score = 0;
            size_t count = n - f - 2;
            for (auto cursor = ranks.get(); count > 0; ++cursor) {
                auto index = *cursor;
                if (pos_to_gradid[2 * index] == i || pos_to_gradid[2 * index + 1] == i) { // Associated distance concerns current gradient
                    score += distances[index];
                    --count;
                }
            }
            scores[i] = score;
        }
    }
    { // Select the 'm' smallest scoring gradients and average them
        auto selected = vlarray<size_t>(n); // Index of the selected gradients
        { // Compute 'selected'
            for (size_t i = 0; i < n; ++i)
                selected[i] = i;
            ::std::nth_element(selected.get(), selected.get() + m, selected.get() + n, [&](size_t a, size_t b) {
                auto&& x = scores[a];
                if (unlikely(!::std::isfinite(x)))
                    return false;
                auto&& y = scores[b];
                if (unlikely(!::std::isfinite(y)))
                    return true;
                return x < y;
            });
        }
        selection_average<T>(inputs, output, selected.get(), m);
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
::torch::Tensor Krum::aggregate_cpu_float(::std::vector<::torch::Tensor> const& inputs, size_t f, size_t m) {
    return aggregate<float>(inputs, f, m);
}
::torch::Tensor Krum::aggregate_cpu_double(::std::vector<::torch::Tensor> const& inputs, size_t f, size_t m) {
    return aggregate<double>(inputs, f, m);
}
