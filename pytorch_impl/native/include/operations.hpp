/**
 * @file   operations.hpp
 * @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
 *
 * @section LICENSE
 *
 * Copyright © 2018-2019 École Polytechnique Fédérale de Lausanne (EPFL).
 * See LICENSE file.
 *
 * @section DESCRIPTION
 *
 * Implementation of common operations.
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
#include "common.hpp"
#include "pytorch.hpp"
#include "threadpool.hpp"

// -------------------------------------------------------------------------- //
// CPU computation helpers

/** Sum-reduction of coordinate-wise squared-difference of two vectors.
 * @param tx Input vector X
 * @param ty Input vector Y
 * @return Output scalar
**/
template<class T> static T reduce_sum_squared_difference(::torch::Tensor const& tx, ::torch::Tensor const& ty) {
    auto const x = tx.data_ptr<T>();
    auto const y = ty.data_ptr<T>();
    ::std::atomic<T> asum{0};
    parallel_for(pool, 0, tx.size(0), [&](size_t start, size_t stop) {
        T sum = 0;
        for (size_t i = start; i < stop; ++i) { // Coordinates to work on
            auto d = x[i] - y[i];
            sum += d * d;
        }
        T tot = 0;
        while (!asum.compare_exchange_strong(tot, tot + sum, ::std::memory_order_relaxed));
    });
    return asum.load(::std::memory_order_relaxed);
}

/** Selection vector arithmetic mean.
 * @param g Non-empty list of vectors of same length
 * @param o Output vector/array of same length
 * @param d Dimension of the vectors (can be automatically deduced)
 * @param a List of indexes to vectors to average
 * @param s Size of the list of indexes
**/
template<class T> static void selection_average(T const* const* g, T* o, size_t d, size_t const* a, size_t s) {
    parallel_for(pool, 0, d, [&](size_t start, size_t stop) {
        for (size_t i = start; i < stop; ++i) { // Coordinates to work on
            T sum = 0;
            for (size_t j = 0; j < s; ++j)
                sum += g[a[j]][i];
            o[i] = sum / static_cast<T>(s);
        }
    });
}
template<class T> static void selection_average(::std::vector<::torch::Tensor> const& g, ::torch::Tensor& o, size_t const* a, size_t s) {
    auto ptrs = vlarray<T const*>(g.size());
    for (size_t i = 0; i < g.size(); ++i)
        ptrs[i] = g[i].data_ptr<T>();
    selection_average(ptrs.get(), o.data_ptr<T>(), o.size(0), a, s);
}
