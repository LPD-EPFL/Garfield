/**
 * @file   bulyan.cpp
 * @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
 *
 * @section LICENSE
 *
 * Copyright © 2018-2019 École Polytechnique Fédérale de Lausanne (EPFL).
 * All rights reserved.
 *
 * @section DESCRIPTION
 *
 * Bulyan over Multi-Krum GAR, native C++ implementation.
 *
 * Based on the algorithm introduced in the following paper:
 *   El Mhamdi El Mahdi, Guerraoui Rachid, and Rouault Sébastien.
 *   The Hidden Vulnerability of Distributed Learning in Byzantium.
 *   In Dy, J. and Krause, A. (eds.), Proceedings of the 35th International
 *   Conference on Machine Learning, volume 80 of Proceedings of Machine
 *   Learning  Research, pp. 3521-3530, Stockholmsmässan, Stockholm Sweden,
 *   10-15 Jul 2018. PMLR. URL http://proceedings.mlr.press/v80/mhamdi18a.html.
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
#include <array.hpp>
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
    // Initialization
    auto const n  = inputs.size();
    auto const d  = static_cast<size_t>(inputs[0].size(0)); // Cast is correct, guaranteed by interface
    auto const kn = n * (n - 1);
    auto const ln = kn / 2;
    auto const t  = n - 2 * f - 2;
    auto const b  = t - 2 * f;
    auto output = ::torch::empty_like(inputs[0]);
    auto output_data = output.data_ptr<T>();
    // Process
    auto distances = vlarray<T>(n * n); // With 'distance[i * n + j]' representing the distance between vectors i & j from score of vector i
    auto scores    = vlarray<T>(n);
    auto ranks     = vlarray<size_t>(ln); // Indexes for 'distances'/'scores', so that 'distances[ranks[i]]'/'scores[ranks[i]]' increases with 'i' ('nan' is treated as '+inf')
    auto intermed  = vlarray<T>(d * t);
    auto inters    = intermed.get();
    { // Initial Krum pass
        auto flat_dist     = vlarray<T>(ln);
        auto pos_to_gradid = vlarray<size_t>(kn);
        { // Distance computations
            auto dstcur = flat_dist.get();
            auto poscur = pos_to_gradid.get();
            for (size_t i = 0; i < n - 1; ++i) {
                distances[i * (n + 1)] = ::std::numeric_limits<T>::max();
                for (size_t j = i + 1; j < n; ++j) {
                    auto dist = reduce_sum_squared_difference<T>(inputs[i], inputs[j]);
                    distances[i * n + j] = dist;
                    distances[j * n + i] = dist;
                    *(dstcur++) = dist;
                    *(poscur++) = i;
                    *(poscur++) = j;
                }
            }
            distances[n * n - 1] = ::std::numeric_limits<T>::max();
        }
        { // Initial score computations and distance pruning
            { // Compute 'ranks'
                for (size_t i = 0; i < ln; ++i)
                    ranks[i] = i;
                ::std::sort(ranks.get(), ranks.get() + ln, [&](size_t a, size_t b) {
                    auto&& x = flat_dist[a];
                    if (unlikely(!::std::isfinite(x)))
                        return false;
                    auto&& y = flat_dist[b];
                    if (unlikely(!::std::isfinite(y)))
                        return true;
                    return x < y;
                });
            }
            parallel_for(pool, 0, n, [&](size_t start, size_t stop) {
                for (size_t i = start; i < stop; ++i) {
                    // Score computation
                    T score = 0;
                    size_t count = n - f - 2;
                    auto cursor = ranks.get();
                    for (; count > 0; ++cursor) {
                        auto index = *cursor;
                        if (pos_to_gradid[2 * index] == i || pos_to_gradid[2 * index + 1] == i) { // Associated distance concerns current gradient
                            score += flat_dist[index];
                            --count;
                        }
                    }
                    scores[i] = score;
                    // Distance pruning
                    count += f + 1;
                    for (; count > 0; ++cursor) {
                        auto index = *cursor;
                        auto a = pos_to_gradid[2 * index];
                        auto b = pos_to_gradid[2 * index + 1];
                        if (a == i) { // Associated distance concerns current gradient
                            distances[i * n + b] = 0;
                            --count;
                        } else if (b == i) { // Associated distance concerns current gradient
                            distances[i * n + a] = 0;
                            --count;
                        }
                    }
                }
            });
        }
    }
    { // Selection loop
        for (size_t i = 0; i < n; ++i) // Initialize 'ranks'
            ranks[i] = i;
        auto inputs_data = vlarray<T const*>(inputs.size());
        for (size_t i = 0; i < inputs.size(); ++i)
            inputs_data[i] = inputs[i].data_ptr<T>();
        for (size_t k = 0;;) {
            // Compute ranks
            ::std::sort(ranks.get(), ranks.get() + n, [&](size_t a, size_t b) {
                auto&& x = scores[a];
                if (unlikely(!::std::isfinite(x)))
                    return false;
                auto&& y = scores[b];
                if (unlikely(!::std::isfinite(y)))
                    return true;
                return x < y;
            });
            // Average the 'm - k' smallest-scoring gradients as the output of Krum
            selection_average<T>(inputs_data.get(), inters + k * d, d, ranks.get(), m - k);
            if (++k >= t) // Check if done
                break;
            { // Remove the smallest-scoring gradient
                auto id = ranks[0];
                scores[id] = ::std::numeric_limits<T>::max(); // Virtually remove the gradient from selection
                for (size_t i = 0; i < n; ++i) { // Update the scores
                    if (i == id)
                        continue;
                    scores[i] -= distances[i * n + id]; // Valid since distances have been pruned
                }
            }
        }
    }
    // Averaged-median coordinate-by-coordinate
    Array<Array<T>> grads{inters, {t, d}};
    parallel_for(pool, 0, d, [&](size_t start, size_t stop) {
        for (size_t x = start; x < stop; ++x) { // Coordinates to work on
            typename decltype(grads)::Iter axis, aend;
            ::std::tie(axis, aend) = grads.axis(0, x);
            auto length = aend - axis;
            auto median = axis + length / 2ul;
            ::std::nth_element(axis, median, aend);
            auto zero = *median;
            ::std::nth_element(axis, axis + b, aend, [&](T x, T y) {
                auto dx = x - zero;
                if (dx < 0)
                    dx = -dx;
                auto dy = y - zero;
                if (dy < 0)
                    dy = -dy;
                return dx < dy;
            });
            auto average = axis[0];
            for (size_t i = 1; i < b; ++i)
                average += axis[i];
            output_data[x] = average / static_cast<T>(b);
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
::torch::Tensor Bulyan::aggregate_cpu_float(::std::vector<::torch::Tensor> const& inputs, size_t f, size_t m) {
    return aggregate<float>(inputs, f, m);
}
::torch::Tensor Bulyan::aggregate_cpu_double(::std::vector<::torch::Tensor> const& inputs, size_t f, size_t m) {
    return aggregate<double>(inputs, f, m);
}
