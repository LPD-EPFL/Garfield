/**
 * @file   bulyan.cu
 * @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
 *
 * @section LICENSE
 *
 * Copyright © 2018-2019 École Polytechnique Fédérale de Lausanne (EPFL).
 *
 * @section DESCRIPTION
 *
 * Bulyan over Multi-Krum GAR, native CUDA implementation.
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
constexpr size_t nb_threads_per_warp  = 32;
constexpr size_t nb_items_per_thread  = 1;
constexpr size_t max_shared_per_block = 48 * 1024;

/** Get the number of blocks needed to process the entries with the current configuration.
 * @param nb_items Number of items to process
 * @return Required number of blocks
**/
constexpr size_t nb_blocks(size_t nb_items) {
    return (nb_items + nb_items_per_thread * nb_threads_per_block - 1) / (nb_items_per_thread * nb_threads_per_block);
}

/** Finalize the initialization of the distances.
 * @param flat_dist Input/output distance array
 * @param distances Output distance matrix
 * @param pos_to_id Output vector
 * @param n         Number of vectors
**/
template<class T> __global__ void distances_finalize(T* flat_dist, T* distances, size_t* pos_to_id, size_t n) {
    size_t i = blockIdx.x / n;
    size_t j = blockIdx.x % n;
    if (i == j) {
        distances[i * (n + 1)] = ::std::numeric_limits<T>::infinity();
    } else if (i < j) {
        auto pos = (2ul * n - i - 3ul) * i / 2ul + j - 1ul; // Magic formula
        auto dist = flat_dist[pos];
        if (!::std::isfinite(dist)) { // Early replacement of non-finite values
            dist = ::std::numeric_limits<T>::infinity();
            flat_dist[pos] = dist;
        }
        distances[i * n + j] = dist;
        distances[j * n + i] = dist;
        pos_to_id[2ul * pos + 0ul] = i;
        pos_to_id[2ul * pos + 1ul] = j;
    } // Else do nothing
}

/** Special single-block merge-sort for indirected array of limited length.
 * @param value  Input constant value array
 * @param rank   (Input/)output only "rank" array
 * @param length Length of the arrays (at most 2 × #threads/block)
**/
template<class T> __global__ void merge_sort_limited(T const* value, size_t* rank, size_t length) {
    // Shared variable declaration
    __shared__ size_t ranks[2][2 * nb_threads_per_block]; // Allocate maximum size
    static_assert(sizeof(ranks) <= max_shared_per_block, "Allocating more shared memory than (supposedly) available");
    // Initialize first selected rank array
    for (size_t i = threadIdx.x; i < length; i += blockDim.x)
        ranks[0][i] = i;
    __syncthreads();
    // Compute thread position to maximize spreading over the available warps
    size_t const id = threadIdx.x / nb_threads_per_warp + (nb_threads_per_block / nb_threads_per_warp) * (threadIdx.x % nb_threads_per_warp);
    // Merge-sort main loop
    size_t selrank = 0; // Currently selected rank array
    size_t level = 1;
    while (level < length) {
        auto const rankend = &(ranks[selrank][length]);
        auto const othrank = (selrank + 1) % 2;
        auto const nextlvl = level << 1;
        auto const offset  = id * nextlvl; // Offset in the array
        if (offset < length) { // This thread has work to do
            auto* bl = &(ranks[selrank][offset]);
            auto* br = bl + level;
            if (br - rankend > 0)
                br = rankend;
            auto* blm = br;
            auto* brm = br + level;
            if (brm - rankend > 0)
                brm = rankend;
            auto* bo = &(ranks[othrank][offset]);
            auto ownlen = &(ranks[othrank][length]) - bo;
            if (ownlen > nextlvl)
                ownlen = nextlvl;
            for (size_t i = 0; i < ownlen; ++i) {
                if (br == brm) { // Select left as right is empty
                    *(bo++) = *(bl++);
                } else if (bl == blm) { // Select right as left is empty
                    *(bo++) = *(br++);
                } else {
                    if (value[*bl] <= value[*br]) { // Select left
                        *(bo++) = *(bl++);
                    } else { // Select right
                        *(bo++) = *(br++);
                    }
                }
            }
        }
        level = nextlvl;
        selrank = othrank;
        __syncthreads();
    }
    // Write back selected rank array
    for (size_t i = threadIdx.x; i < length; i += blockDim.x)
        rank[i] = ranks[selrank][i];
}

/** Compute the scores and prune the distances for all the gradients.
 * @param flat_dist Input constant distances array
 * @param ranks     Input constant distance ordering array
 * @param pos_to_id Map between position in 'flat_dist' to pair of corresponding gradient ID
 * @param n         Total number of gradients (at most #threads/block)
 * @param f         Number of byzantine workers to tolerate (must be < n)
 * @param scores    Output score array
 * @param distances Output distance matrix to prune
**/
template<class T> __global__ void compute_scores_prune_distances(T const* flat_dist, size_t const* ranks, size_t const* pos_to_id, size_t n, size_t f, T* scores, T* distances) {
    // Compute thread position to maximize spreading over the available warps
    size_t const id = threadIdx.x / nb_threads_per_warp + (nb_threads_per_block / nb_threads_per_warp) * (threadIdx.x % nb_threads_per_warp);
    if (id >= n) // Nothing to do
        return;
    // Score computation
    T score = 0;
    size_t count = n - f - 2;
    auto cursor = ranks;
    for (; count > 0; ++cursor) {
        auto index = *cursor;
        if (pos_to_id[2 * index] == id || pos_to_id[2 * index + 1] == id) { // Associated distance concerns current gradient
            score += flat_dist[index];
            --count;
        }
    }
    if (!::std::isfinite(score)) // Replacement of non-finite values
        score = ::std::numeric_limits<T>::infinity();
    scores[id] = score;
    // Distance pruning
    count += f + 1;
    for (; count > 0; ++cursor) {
        auto index = *cursor;
        auto a = pos_to_id[2 * index];
        auto b = pos_to_id[2 * index + 1];
        if (a == id) { // Associated distance concerns current gradient
            distances[id * n + b] = 0;
            --count;
        } else if (b == id) { // Associated distance concerns current gradient
            distances[id * n + a] = 0;
            --count;
        }
    }
}

/** Simply removes the smallest scoring gradient from the scores of the others.
 * @param scores    Input/output score array
 * @param distances Input constant pruned distance matrix
 * @param ranks     Output only "rank" array
 * @param n         Length of the arrays (at most 2 × #threads/block)
**/
template<class T> __global__ void remove_smallest_scoring_gradient(T* scores, T const* distances, size_t const* ranks, size_t n) {
    // Compute thread position to maximize spreading over the available warps
    size_t const id = threadIdx.x / nb_threads_per_warp + (nb_threads_per_block / nb_threads_per_warp) * (threadIdx.x % nb_threads_per_warp);
    if (id >= n) // Nothing to do
        return;
    // Update the scores
    auto gid = ranks[0];
    if (id == gid) {
        scores[id] = ::std::numeric_limits<T>::infinity(); // Virtually remove the gradient from selection
    } else {
        scores[id] -= distances[id * n + gid]; // Valid since distances have been pruned
    }
}

/** Output for each coordinate the median of the input coordinate(s).
 * @param inputs Input gradient matrix (can be modified)
 * @param output Output aggregated gradient
 * @param d      Gradient space dimension
 * @param n      Number of input gradients
**/
template<class T> __global__ void simple_median(T* inputs, T* output, size_t d, size_t t) {
    for (size_t x = blockIdx.x * blockDim.x + threadIdx.x; x < d; x += blockDim.x * gridDim.x) // For each coordinate i
        output[x] = partition_pivot<T>(inputs + x, inputs + x + d * (t / 2), t, d);
}

/** Compute for each coordinate the average of the 'b' closest coordinate(s) to the median.
 * @param inputs Input gradient matrix (can be modified)
 * @param output Output aggregated gradient
 * @param d      Gradient space dimension
 * @param n      Number of input gradients
 * @param b      Number of coordinates to average
**/
template<class T> __global__ void averaged_median(T* inputs, T* output, size_t d, size_t t, size_t b) {
    for (size_t x = blockIdx.x * blockDim.x + threadIdx.x; x < d; x += blockDim.x * gridDim.x) { // For each coordinate i
        auto axis = inputs + x;
        // Find the median
        auto median = partition_pivot<T>(axis, axis + d * (t / 2), t, d);
        // Partition with the beta closest
        partition_pivot<T>(axis, axis + d * b, t, d, [=](T x) {
            x -= median;
            return sqrt(x * x);
        });
        // Average the b closest coordinate(s) to the median
        auto average = *(axis + 0);
        for (size_t i = 1; i < b; ++i)
            average += *(axis + d * i);
        output[x] = average / static_cast<T>(b);
    }
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
    auto const d  = inputs[0].size(0);
    auto const kn = n * (n - 1);
    auto const ln = kn / 2;
#ifndef NDEBUG
    if (unlikely(ln > 2 * nb_threads_per_block))
        AT_ERROR("Need n * (n - 1) / 2 <= ", 2 * nb_threads_per_block, ", got ", ln); // Implies n <= nb_threads_per_block (for n >= 5, which is true)
#endif
    auto const t = n - 2 * f - 2;
    auto const b = t - 2 * f;
    cudaStream_t stream = 0; // Default stream
    size_t lr;
    CUDA_ASSERT(cub::DeviceReduce::Sum(nullptr, lr, static_cast<T const*>(nullptr), static_cast<T*>(nullptr), d, stream));
    auto temp0 = vlcudarray<T>(n * n + ln + n + lr + d * t); // Distances + flat distances + reduction vector + [ intermediate vector | selected vectors ]
    auto temp1 = vlcudarray<size_t>(ln + kn); // [ Ranks | index of the selected gradients ] + position to gradient ID
    CUDArray<T const*> gpu_inputs{inputs, [](::torch::Tensor const& elem) -> T const* { return elem.data_ptr<T>(); }};
    auto output = ::torch::empty_like(inputs[0]);
    auto const gpu_output    = output.data_ptr<T>();
    auto const gpu_distances = temp0.get();
    auto const gpu_flat_dist = gpu_distances + n * n;
    auto const gpu_scores    = gpu_flat_dist + ln;
    auto const gpu_reduce    = gpu_scores + n;
    auto const gpu_inters    = gpu_reduce + lr;
    auto const gpu_ranks     = temp1.get();
    auto const gpu_pos_to_id = gpu_ranks + ln;
    // Process
    { // Initial Krum pass - distance computations
        auto dstcur = gpu_flat_dist;
        for (size_t i = 0; i < n - 1; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                squared_difference<T><<<nb_blocks(d), nb_threads_per_block, 0, stream>>>(inputs[i].data_ptr<T>(), inputs[j].data_ptr<T>(), gpu_inters, d);
                CUDA_ASSERT_KERNEL();
                CUDA_ASSERT(cub::DeviceReduce::Sum(gpu_reduce, lr, gpu_inters, dstcur, d, stream));
                ++dstcur;
            }
        }
        distances_finalize<T><<<n * n, 1, 0, stream>>>(gpu_flat_dist, gpu_distances, gpu_pos_to_id, n);
        CUDA_ASSERT_KERNEL();
    }
    // Initial score computations and distance pruning
    merge_sort_limited<T><<<1, nb_threads_per_block, 0, stream>>>(gpu_flat_dist, gpu_ranks, ln); // Compute 'gpu_ranks' for 'gpu_flat_dist'
    CUDA_ASSERT_KERNEL();
    compute_scores_prune_distances<T><<<1, nb_threads_per_block, 0, stream>>>(gpu_flat_dist, gpu_ranks, gpu_pos_to_id, n, f, gpu_scores, gpu_distances);
    CUDA_ASSERT_KERNEL();
    // Krum optimized selection loop
    for (size_t k = 0;;) {
        // Compute ranks
        merge_sort_limited<T><<<1, nb_threads_per_block, 0, stream>>>(gpu_scores, gpu_ranks, n); // Compute 'gpu_ranks' for 'gpu_scores'
        CUDA_ASSERT_KERNEL();
        // Average the 'm - k' smallest-scoring gradients as the output of Krum
        selection_average<T><<<nb_blocks(d), nb_threads_per_block, 0, stream>>>(gpu_inputs.data(), gpu_inters + k * d, d, reinterpret_cast<size_t*>(gpu_ranks), m - k);
        CUDA_ASSERT_KERNEL();
        // Check if done
        if (++k >= t)
            break;
        // Remove the smallest-scoring gradient
        remove_smallest_scoring_gradient<T><<<1, nb_threads_per_block, 0, stream>>>(gpu_scores, gpu_distances, gpu_ranks, n);
        CUDA_ASSERT_KERNEL();
    }
    // Averaged-median coordinate-by-coordinate
    averaged_median<T><<<nb_blocks(d), nb_threads_per_block, 0, stream>>>(gpu_inters, gpu_output, d, t, b);
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
::torch::Tensor Bulyan::aggregate_gpu_float(::std::vector<::torch::Tensor> const& inputs, size_t f, size_t m) {
    return aggregate<float>(inputs, f, m);
}
::torch::Tensor Bulyan::aggregate_gpu_double(::std::vector<::torch::Tensor> const& inputs, size_t f, size_t m) {
    return aggregate<double>(inputs, f, m);
}
