/**
 * @file   operations.cu.hpp
 * @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
 *
 * @section LICENSE
 *
 * Copyright © 2018-2019 École Polytechnique Fédérale de Lausanne (EPFL).
 * All rights reserved.
 *
 * @section DESCRIPTION
 *
 * CUDA kernels for common operations.
**/

#pragma once

// Compiler version check
#ifndef __CUDACC__
    #error This translation unit requires a CUDA compiler
#endif
#if __cplusplus < 201103L
    #error This translation unit requires at least a C++11 compiler
#endif
#ifndef __GNUC__
    #error This translation unit requires a GNU C++ compiler
#endif

// External headers
#include <cub/cub.cuh>

// Internal headers
#include "common.hpp"

// -------------------------------------------------------------------------- //
// GPU computation helpers

/** Coordinate-wise squared-difference of two vectors.
 * @param x Input vector X
 * @param y Input vector Y
 * @param o Output vector
 * @param d Vector space dimension
**/
template<class T> __global__ static void squared_difference(T const* x, T const* y, T* o, size_t d) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < d; i += blockDim.x * gridDim.x) {
        auto r = x[i] - y[i];
        o[i] = r * r;
    }
}

/** Selection vector arithmetic mean.
 * @param g List of vectors
 * @param o Output vector
 * @param d Vector space dimension
 * @param a List of indexes to vectors to average
 * @param s Size of the list of indexes
**/
template<class T> __global__ static void selection_average(T const* const* g, T* o, size_t d, size_t const* a, size_t s) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < d; i += blockDim.x * gridDim.x) {
        T sum = 0;
        for (size_t j = 0; j < s; ++j)
            sum += g[a[j]][i];
        o[i] = sum / static_cast<T>(s);
    }
}

/** Partition in-place the given (small) array, does not support non finite inputs.
 * @param values Input/output value array
 * @param pivot  Position used as the pivot
 * @param length Array length (in number of steps)
 * @param step   Array step
 * @param conv   Conversion function (optional)
 * @return Value at pivot after partitioning
**/
template<class T, class Conv> __device__ static T partition_pivot(T* values, T* const pivot, size_t length, size_t step, Conv&& conv) {
    auto       top   = values;
    auto       bot   = values + step * (length - 1);
    auto const mid   = (pivot == bot ? bot - step : (pivot == top ? top + step : pivot));
    auto       max   = ::std::numeric_limits<T>::lowest(); // Max of the mins
    auto       min   = ::std::numeric_limits<T>::max(); // Min of the maxs
    T*         pmax  = nullptr; // Position of the current max
    T*         pmin  = nullptr; // Position of the current min
    if (length > 2) while (true) {
        // Triple exchange, without branching
        T   v[] = {*top, *mid, *bot};
        T   w[] = {conv(v[0]), conv(v[1]), conv(v[2])};
        int c[] = {w[0] > w[1], w[0] > w[2], w[1] > w[2]};
        int p[] = {(1 + c[0] + 2 * c[1] + c[2] - (c[1] ^ c[2])) / 2, (4 - c[0] - 2 * c[1] - c[2] + (c[0] ^ c[1])) / 2}; // Magic formulas
        T   d[] = {w[p[0]], w[3 - p[0] - p[1]], w[p[1]]};
        *top = v[p[0]];
        *mid = v[3 - p[0] - p[1]];
        *bot = v[p[1]];
        // Update max of mins and min of maxs
        if (d[0] > max) {
            max  = d[0];
            pmax = top;
        }
        if (d[2] < min) {
            min  = d[2];
            pmin = bot;
        }
        // Move top and bottom cursors
        size_t done = 0;
        if (max > d[1]) { // Rollback pointer
            top = (pmax == top ? values : pmax); // Assume worse case when the current maximum may have been swapped
            max = ::std::numeric_limits<T>::lowest();
        } else {
            auto ntop = top + step;
            if (ntop != mid) {
                top = ntop;
            } else {
                ++done;
            }
        }
        if (min < d[1]) { // Rollback pointer
            bot = (pmin == bot ? values + step * (length - 1) : pmin); // Assume worse case when the current minimum may have been swapped
            min = ::std::numeric_limits<T>::max();
        } else {
            auto nbot = bot - step;
            if (nbot != mid) {
                bot = nbot;
            } else {
                ++done;
            }
        }
        // Check if done
        if (done >= 2)
            break;
    } else if (length == 2) {
        T v[] = {*top, *bot};
        if (conv(v[0]) > conv(v[1])) {
            *top = v[1];
            *bot = v[0];
        } // Else already sorted
    } // Else already sorted
    return *pivot;
}
template<class T> __device__ static T partition_pivot(T* values, T* const pivot, size_t length, size_t step) {
    return partition_pivot(values, pivot, length, step, [](T x) { return x; });
}
