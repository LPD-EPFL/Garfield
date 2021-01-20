/**
 * @file   gpu.cu.cpp
 * @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
 *
 * @section LICENSE
 *
 * Copyright © 2018-2019 École Polytechnique Fédérale de Lausanne (EPFL).
 * See LICENSE file.
 *
 * @section DESCRIPTION
 *
 * Multi-Krum GAR, GPU CUDA kernel implementation.
 *
 * Based on the algorithm introduced in the following paper:
 *   Blanchard Peva, El Mhamdi El Mahdi, Guerraoui Rachid, and Stainer Julien.
 *   Machine learning with adversaries: Byzantine tolerant gradient descent.
 *   In Advances in Neural Information Processing Systems 30, pp.118–128.
 *   Curran Associates, Inc., 2017.
**/

#ifdef GOOGLE_CUDA

#include <algorithm>
#include <cmath>

#define EIGEN_USE_GPU
#include <tensorflow/core/util/cuda_kernel_helper.h>
#include <third_party/eigen3/unsupported/Eigen/CXX11/Tensor>

#include <cub/cub.cuh>

#include <common.hpp>
#include <operations.cu.hpp>
#include "decl.hpp"

// -------------------------------------------------------------------------- //
// GPU kernel configuration and helpers
namespace OP_NAME {

constexpr static size_t nb_threads_per_block = 128;
constexpr static size_t nb_items_per_thread  = 1;

/** Get the number of blocks needed to process the entries with the current configuration.
 * @param nb_items Number of items to process
 * @return Required number of blocks
**/
constexpr static size_t nb_blocks(size_t nb_items) {
    return (nb_items + nb_items_per_thread * nb_threads_per_block - 1) / (nb_items_per_thread * nb_threads_per_block);
}

}
// -------------------------------------------------------------------------- //
// Kernel implementation
namespace OP_NAME {

template<class T> class Kernel<GPUDevice, T>: public Static {
public:
    static void process(OpKernelContext& context, size_t const n, size_t const f, size_t const d, size_t const m, Tensor const& input, Tensor& output) {
        auto const stream = context.eigen_device<GPUDevice>().stream();
        auto const length_distances = n * (n - 1) / 2;
        size_t length_reduction;
        cub::DeviceReduce::Sum(nullptr, length_reduction, static_cast<T const*>(nullptr), static_cast<T*>(nullptr), d, stream);
        Tensor inter; // Distances + intermediate vector + reduction vector
        OP_REQUIRES_OK(&context, context.allocate_temp(DataTypeToEnum<T>::value, TensorShape{static_cast<long long>(length_distances + d + length_reduction)}, &inter));
        auto data_in   = input.flat<T>().data();
        auto data_out  = output.flat<T>().data();
        auto distances = inter.flat<T>().data();
        auto intergrad = distances + length_distances;
        auto reduction = intergrad + d;
        Tensor select; // Index of the selected gradients
        OP_REQUIRES_OK(&context, context.allocate_temp(DataTypeToEnum<tf_size_t>::value, TensorShape{static_cast<long long>(m)}, &select));
        auto selected = select.flat<tf_size_t>().data();
        auto const cn = n;
        auto const kn = n * (n - 1);
        size_t pos_to_gradid[kn];
        { // Distance computations
            auto cursor = distances;
            auto poscur = pos_to_gradid;
            for (size_t i = 0; i < n - 1; ++i) {
                auto x = data_in + i * d;
                for (size_t j = i + 1; j < n; ++j) {
                    *(poscur++) = i;
                    *(poscur++) = j;
                    auto y = data_in + j * d;
                    squared_difference<T><<<nb_blocks(d), nb_threads_per_block, 0, stream>>>(x, y, intergrad, d);
                    cub::DeviceReduce::Sum(reduction, length_reduction, intergrad, cursor, d, stream);
                    ++cursor;
                }
            }
        }
        T cpu_scores[cn];
        { // Score computations
            T cpu_distances[length_distances];
            cudaMemcpyAsync(cpu_distances, distances, length_distances * sizeof(T), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
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
            cudaMemcpyAsync(selected, cpu_selected, m * sizeof(size_t), cudaMemcpyHostToDevice, stream);
            selection_average<T><<<nb_blocks(d), nb_threads_per_block, 0, stream>>>(data_in, data_out, d, reinterpret_cast<size_t*>(selected), m);
            cudaStreamSynchronize(stream); // FIXME: Really needed?
        }
    }
};

// Explicit instantiations
template class Kernel<GPUDevice, float>;
template class Kernel<GPUDevice, double>;

}

#endif
