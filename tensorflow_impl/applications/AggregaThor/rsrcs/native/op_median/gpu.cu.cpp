/**
 * @file   gpu.cu.cpp
 * @author Sébastien Rouault <sebastien.rouault@epfl.ch>
 *
 * @section LICENSE
 *
 * Copyright © 2018-2019 Sébastien ROUAULT.
 *
 * @section DESCRIPTION
 *
 * Median GAR, GPU CUDA kernel implementation.
**/

#ifdef GOOGLE_CUDA

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>

#define EIGEN_USE_GPU
#include <tensorflow/core/util/cuda_kernel_helper.h>
#include <third_party/eigen3/unsupported/Eigen/CXX11/Tensor>

#include <cub/cub.cuh>

#include <common.hpp>
#include <array.hpp>
#include <operations.cu.hpp>
#include "decl.hpp"

// -------------------------------------------------------------------------- //
// GPU kernel configuration and helpers
namespace OP_NAME {

constexpr static size_t nb_threads_per_block = 128;
constexpr static size_t nb_items_per_thread  = 1;

/** Equivalent TF-compatible type for 'size_t'.
**/
using tf_size_t = typename ::std::conditional<
    sizeof(size_t) == sizeof(uint32) && alignof(size_t) == alignof(uint32), uint32, typename ::std::conditional<
    sizeof(size_t) == sizeof(uint64) && alignof(size_t) == alignof(uint64), uint64, void>::type>::type; // Else unsupported target machine, 'void' to trigger a diagnostic

/** Get the number of blocks needed to process the entries with the current configuration.
 * @param nb_items Number of items to process
 * @return Required number of blocks
**/
constexpr static size_t nb_blocks(size_t nb_items) {
    return (nb_items + nb_items_per_thread * nb_threads_per_block + 1) / (nb_items_per_thread * nb_threads_per_block);
}

}
// -------------------------------------------------------------------------- //
// GPU implementations
namespace OP_NAME {

constexpr static size_t max_n = 1024;

/** Compute for each coordinate the average of the 'b' closest coordinate(s) to the median.
 * @param inputs Input gradient matrix
 * @param output Output aggregated gradient
 * @param n      Number of input gradients, must be below 'max_n'
 * @param d      Gradient space dimension
**/
template<class T> __global__ static void median_nan(T const* inputs, T* output, size_t n, size_t d) {
    T copy[max_n];
    for (size_t x = blockIdx.x * blockDim.x + threadIdx.x; x < d; x += blockDim.x * gridDim.x) { // For each coordinate i
        T const* axis = inputs + x;
        size_t length = n;
        // Finite value-only copying
        T* target = copy;
        for (size_t i = 0; i < length; ++i) {
            T value = *(axis + d * i);
            if (::std::isfinite(value)) {
                *(target++) = value;
            } else {
                --length;
            }
        }
        // Median coordinate-wise
        output[x] = partition_pivot<T>(copy, copy + length / 2ul, length, 1);
    }
}

}
// -------------------------------------------------------------------------- //
// Kernel implementation
namespace OP_NAME {

template<class T> class Kernel<GPUDevice, T>: public Static {
public:
    static void process(OpKernelContext& context, size_t const n, size_t const d, Tensor const& input, Tensor& outpt) {
        OP_REQUIRES(&context, n <= max_n, errors::InvalidArgument("Median (GPU) can aggregate at most n = ", max_n, " gradients, got ", n));
        auto const stream = context.eigen_device<GPUDevice>().stream();
        auto const gpu_inputs = input.flat<T>().data();
        auto const gpu_output = outpt.flat<T>().data();
        // Averaged-median coordinate-by-coordinate
        median_nan<T><<<nb_blocks(d), nb_threads_per_block, 0, stream>>>(gpu_inputs, gpu_output, n, d);
    }
};

// Explicit instantiations
template class Kernel<GPUDevice, float>;
template class Kernel<GPUDevice, double>;

}

#endif
