/**
 * @file   median.cu
 * @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
 *
 * @section LICENSE
 *
 * Copyright © 2018-2019 École Polytechnique Fédérale de Lausanne (EPFL).
 * See LICENSE file.
 *
 * @section DESCRIPTION
 *
 * Non-finite-proof median GAR, native CUDA implementation.
**/

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
#include <algorithm>
#include <cmath>
#include <cub/cub.cuh>
#include <limits>
#include <vector>

// Internal headers
#include <common.hpp>
#include <cudarray.cu.hpp>
#include <operations.cu.hpp>
#include <pytorch.hpp>
#include "rule.hpp"

// -------------------------------------------------------------------------- //
// CUDA kernel templates
namespace {

constexpr size_t nb_threads_per_block = 128;
constexpr size_t nb_items_per_thread  = 1;

/** Get the number of blocks needed to process the entries with the current configuration.
 * @param nb_items Number of items to process
 * @return Required number of blocks
**/
constexpr size_t nb_blocks(size_t nb_items) {
    return (nb_items + nb_items_per_thread * nb_threads_per_block - 1) / (nb_items_per_thread * nb_threads_per_block);
}

/** Non-finite-proof coordinate-wise median computation, expect per-thread shared memory of size 'n * sizeof(T)'.
 * @param inputs Input gradients
 * @param output Output gradient
 * @param n      Number of input gradients
 * @param d      Gradient space dimension
**/
template<class T> __global__ static void median(T const* const* inputs, T* output, size_t n, size_t d) {
    // Initialization
    extern __shared__ __align__(sizeof(T)) char s[];
    auto const id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= d) // Nothing to do
        return;
    auto* local = reinterpret_cast<T*>(s) + n * threadIdx.x; // In shared memory but used locally
    { // Copy all values to shared memory + make non-finite large-finite + push them to the end
        size_t low = 0;
        size_t top = n;
        for (size_t i = 0; i < n; ++i) {
            auto val = inputs[i][id];
            if (::std::isfinite(val)) {
                local[low++] = val;
            } else {
                local[--top] = ::std::numeric_limits<T>::max();
            }
        }
    }
    // Median coordinate-wise
    partition_pivot(local, local + n / 2, n, 1); // Use same pivot to avoid divergence
    // Write median
    output[id] = local[n / 2];
}

}
// -------------------------------------------------------------------------- //
// Native implementation
namespace {

/** Aggregate the given tensor(s).
 * @param inputs Given n tensors(s), at least one, must all be continuous 1-dim tensor(s) of same (non-null) size, data type and device
 * @return Aggregated 1-dim tensor of same size, data type and device
**/
template<class T> ::torch::Tensor aggregate(::std::vector<::torch::Tensor> const& inputs) {
    // Initialize
    cudaStream_t stream = 0; // Default stream
    auto const n = inputs.size();
    auto const d = inputs[0].size(0);
    auto output = ::torch::empty_like(inputs[0]);
    // Process
    CUDArray<T const*> inputs_array{inputs.data(), inputs.size(), [](::torch::Tensor const& elem) -> T const* { return elem.data_ptr<T>(); }};
    median<T><<<nb_blocks(d), nb_threads_per_block, nb_threads_per_block * n * sizeof(T), stream>>>(inputs_array.data(), output.data_ptr<T>(), n, d);
    CUDA_ASSERT_KERNEL();
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
::torch::Tensor Median::aggregate_gpu_float(::std::vector<::torch::Tensor> const& inputs) {
    return aggregate<float>(inputs);
}
::torch::Tensor Median::aggregate_gpu_double(::std::vector<::torch::Tensor> const& inputs) {
    return aggregate<double>(inputs);
}
