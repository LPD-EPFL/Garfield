/**
 * @file   brute.cu
 * @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
 *
 * @section LICENSE
 *
 * Copyright © 2019 École Polytechnique Fédérale de Lausanne (EPFL).
 * See LICENSE file.
 *
 * @section DESCRIPTION
 *
 * Brute GAR, native CUDA implementation.
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
#include <cub/cub.cuh>
#include <limits>
#include <vector>

// Internal headers
#include <combinations.hpp>
#include <common.hpp>
#include <cudarray.cu.hpp>
#include <operations.cu.hpp>
#include <pytorch.hpp>
#include <threadpool.hpp>
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

}
// -------------------------------------------------------------------------- //
// Native implementation
namespace {

/** Aggregate the given tensor(s).
 * @param inputs Given n tensors(s), at least one, must all be continuous 1-dim tensor(s) of same (non-null) size, data type and device
 * @param f      Number of Byzantine tensor(s) to tolerate, must be positive
 * @return Aggregated 1-dim tensor of same size, data type and device
**/
template<class T> ::torch::Tensor aggregate(::std::vector<::torch::Tensor> const& inputs, size_t f) {
    // Initialize
    auto const n = inputs.size();
    auto const d = inputs[0].size(0);
    auto const length_distances = n * (n - 1) / 2;
    cudaStream_t stream = 0; // Default stream
    size_t length_reduction;
    CUDA_ASSERT(cub::DeviceReduce::Sum(nullptr, length_reduction, static_cast<T const*>(nullptr), static_cast<T*>(nullptr), d, stream));
    auto inter     = vlcudarray<T>(length_distances + d + length_reduction); // Distances + intermediate vector + reduction vector
    auto distances = inter.get();
    auto intergrad = distances + length_distances;
    auto reduction = intergrad + d;
    auto output = ::torch::empty_like(inputs[0]);
    // Process
    { // Distance computations
        auto cursor = distances;
        for (size_t i = 0; i < n - 1; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                squared_difference<T><<<nb_blocks(d), nb_threads_per_block, 0, stream>>>(inputs[i].data_ptr<T>(), inputs[j].data_ptr<T>(), intergrad, d);
                CUDA_ASSERT_KERNEL();
                CUDA_ASSERT(cub::DeviceReduce::Sum(reduction, length_reduction, intergrad, cursor, d, stream));
                ++cursor;
            }
        }
    }
    size_t global_position;
    { // Smallest-diameter subgroup parallel selection
        auto cpu_distances = vlarray<T>(length_distances);
        CUDA_ASSERT(cudaMemcpyAsync(cpu_distances.get(), distances, length_distances * sizeof(T), cudaMemcpyDeviceToHost, stream));
        CUDA_ASSERT(cudaStreamSynchronize(stream));
        { // Convert non-finite distances to max distance
            for (size_t i = 0; i < length_distances; ++i) {
                auto value = cpu_distances[i];
                if (!::std::isfinite(value))
                    cpu_distances[i] = ::std::numeric_limits<T>::max();
            }
        }
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
                        auto const dist = cpu_distances[(2 * n - x - 3) * x / 2 + y - 1];
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
        CUDArray<T const*> inputs_array{inputs.data(), inputs.size(), [](::torch::Tensor const& elem) -> T const* { return elem.data_ptr<T>(); }};
        Combinations selected{n, n - f};
        selected.seek(global_position);
        CUDArray<size_t> selection_array{selected.get_current().data(), selected.get_current().size(), [](Combinations::Index const& elem) -> size_t { return elem; }};
        selection_average<T><<<nb_blocks(d), nb_threads_per_block, 0, stream>>>(inputs_array.data(), output.data_ptr<T>(), d, selection_array.data(), n - f);
        CUDA_ASSERT_KERNEL();
        CUDA_ASSERT(cudaStreamSynchronize(stream)); // FIXME: Really needed?
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
::torch::Tensor Brute::aggregate_gpu_float(::std::vector<::torch::Tensor> const& inputs, size_t f) {
    return aggregate<float>(inputs, f);
}
::torch::Tensor Brute::aggregate_gpu_double(::std::vector<::torch::Tensor> const& inputs, size_t f) {
    return aggregate<double>(inputs, f);
}
