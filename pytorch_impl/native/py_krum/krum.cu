/**
 * @file   krum.cu
 * @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
 *
 * @section LICENSE
 *
 * Copyright © 2018-2019 École Polytechnique Fédérale de Lausanne (EPFL).
 * See LICENSE file.
 *
 * @section DESCRIPTION
 *
 * Multi-Krum GAR, native CUDA implementation.
 *
 * Based on the algorithm introduced in the following paper:
 *   Blanchard Peva, El Mhamdi El Mahdi, Guerraoui Rachid, and Stainer Julien.
 *   Machine learning with adversaries: Byzantine tolerant gradient descent.
 *   In Advances in Neural Information Processing Systems 30, pp.118–128.
 *   Curran Associates, Inc., 2017.
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

}
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
    auto const cn = n;
    auto const kn = n * (n - 1);
    auto const d  = inputs[0].size(0);
    auto const length_distances = kn / 2;
    cudaStream_t stream = 0; // Default stream
    size_t length_reduction;
    CUDA_ASSERT(cub::DeviceReduce::Sum(nullptr, length_reduction, static_cast<T const*>(nullptr), static_cast<T*>(nullptr), d, stream));
    auto inter     = vlcudarray<T>(length_distances + d + length_reduction); // Distances + intermediate vector + reduction vector
    auto distances = inter.get();
    auto intergrad = distances + length_distances;
    auto reduction = intergrad + d;
    auto select = vlcudarray<size_t>(m); // Index of the selected gradients
    auto selected = select.get();
    auto output = ::torch::empty_like(inputs[0]);
    // Process
    auto pos_to_gradid = vlarray<size_t>(kn);
    { // Distance computations
        auto cursor = distances;
        auto poscur = pos_to_gradid.get();
        for (size_t i = 0; i < n - 1; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                squared_difference<T><<<nb_blocks(d), nb_threads_per_block, 0, stream>>>(inputs[i].data_ptr<T>(), inputs[j].data_ptr<T>(), intergrad, d);
                CUDA_ASSERT_KERNEL();
                CUDA_ASSERT(cub::DeviceReduce::Sum(reduction, length_reduction, intergrad, cursor, d, stream));
                ++cursor;
                *(poscur++) = i;
                *(poscur++) = j;
            }
        }
    }
    T cpu_scores[cn];
    { // Score computations
        T cpu_distances[length_distances];
        CUDA_ASSERT(cudaMemcpyAsync(cpu_distances, distances, length_distances * sizeof(T), cudaMemcpyDeviceToHost, stream));
        CUDA_ASSERT(cudaStreamSynchronize(stream));
        size_t cpu_ranks[length_distances]; // Indexes for 'cpu_distances', so that 'cpu_distances[cpu_ranks[i]]' increases with 'i' ('nan' is treated as '+inf')
        { // Compute 'cpu_ranks'
            for (size_t i = 0; i < length_distances; ++i)
                cpu_ranks[i] = i;
            T* cpu_dist_ptr = cpu_distances;
            ::std::sort(cpu_ranks, cpu_ranks + length_distances, [cpu_dist_ptr](size_t a, size_t b) {
                auto&& x = cpu_dist_ptr[a];
                if (unlikely(!::std::isfinite(x)))
                    return false;
                auto&& y = cpu_dist_ptr[b];
                if (unlikely(!::std::isfinite(y)))
                    return true;
                return x < y;
            });
        }
        for (size_t i = 0; i < n; ++i) { // Compute 'scores'
            T score = 0;
            size_t count = n - f - 2;
            for (auto* cursor = cpu_ranks; count > 0; ++cursor) {
                auto index = *cursor;
                if (pos_to_gradid[2 * index] == i || pos_to_gradid[2 * index + 1] == i) { // Associated distance concerns current gradient
                    score += cpu_distances[index];
                    --count;
                }
            }
            cpu_scores[i] = score;
        }
    }
    { // Select the 'm' smallest scoring gradients and average them
        size_t cpu_selected[cn]; // Index of the selected gradients
        { // Compute 'cpu_selected'
            for (size_t i = 0; i < cn; ++i)
                cpu_selected[i] = i;
            T* cpu_scores_ptr = cpu_scores;
            ::std::nth_element(cpu_selected, cpu_selected + m, cpu_selected + n, [cpu_scores_ptr](size_t a, size_t b) {
                auto&& x = cpu_scores_ptr[a];
                if (unlikely(!::std::isfinite(x)))
                    return false;
                auto&& y = cpu_scores_ptr[b];
                if (unlikely(!::std::isfinite(y)))
                    return true;
                return x < y;
            });
        }
        CUDA_ASSERT(cudaMemcpyAsync(selected, cpu_selected, m * sizeof(size_t), cudaMemcpyHostToDevice, stream));
        CUDArray<T const*> inputs_array{inputs.data(), inputs.size(), [](::torch::Tensor const& elem) -> T const* { return elem.data_ptr<T>(); }};
        selection_average<T><<<nb_blocks(d), nb_threads_per_block, 0, stream>>>(inputs_array.data(), output.data_ptr<T>(), d, reinterpret_cast<size_t*>(selected), m);
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
::torch::Tensor Krum::aggregate_gpu_float(::std::vector<::torch::Tensor> const& inputs, size_t f, size_t m) {
    return aggregate<float>(inputs, f, m);
}
::torch::Tensor Krum::aggregate_gpu_double(::std::vector<::torch::Tensor> const& inputs, size_t f, size_t m) {
    return aggregate<double>(inputs, f, m);
}
