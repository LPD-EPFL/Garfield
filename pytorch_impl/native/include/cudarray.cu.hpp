/**
 * @file   cudarray.cu.hpp
 * @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
 *
 * @section LICENSE
 *
 * Copyright © 2018-2019 École Polytechnique Fédérale de Lausanne (EPFL).
 * See LICENSE file.
 *
 * @section DESCRIPTION
 *
 * Simple CPU array/vector to GPU fixed-size array and automatic (de)allocation.
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
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

// Internal headers
#include "common.hpp"

// -------------------------------------------------------------------------- //
// CPU array/vector to GPU fixed-size array

/** CPU array/vector to GPU fixed-size array.
 * @param GPUElem Underlying element type on GPU
**/
template<class GPUElem> class CUDArray final {
    static_assert(!::std::is_reference<GPUElem>::value, "Expected 'GPUElem' not to be a reference");
    static_assert(::std::is_trivially_copyable<GPUElem>::value, "Expected 'GPUElem' to be trivially copyable");
    static_assert(::std::is_trivially_destructible<GPUElem>::value, "Expected 'GPUElem' to be trivially destructible");
private:
    /** This class instance.
    **/
    using This = CUDArray<GPUElem>;
private:
    GPUElem* array; // Array of elements on GPU (nullptr if empty)
    size_t  length; // Size of the array (undefined if empty)
private:
    /** Build the array, must be empty.
     * @param data CPU array data
     * @param size Array size
     * @param conv Conversion function (optional, implicit conversion by default)
    **/
    template<class CPUElem, class Conv> void build(CPUElem const* data, size_t size, Conv&& conv = [](CPUElem const& elem) -> GPUElem { return elem; }) {
        // Allocate data
        auto res = cudaMalloc(&array, sizeof(GPUElem) * size);
        if (unlikely(res != cudaSuccess))
            AT_ERROR("Unable to allocate memory on the GPU: ", cudaGetErrorString(res));
        // Convert and transfer
        try {
            auto buf = vlarray<GPUElem>(size);
            for (size_t i = 0; i < size; ++i)
                new(&(buf[i])) GPUElem{conv(data[i])};
            auto res = ::cudaMemcpy(array, buf.get(), sizeof(GPUElem) * size, cudaMemcpyHostToDevice);
            if (unlikely(res != cudaSuccess))
                AT_ERROR("Unable to copy memory to the GPU: ", cudaGetErrorString(res));
        } catch (...) {
            cudaFree(array);
            throw;
        }
        // Finalization
        length = size;
    }
    /** Clear the array, if not empty.
    **/
    void clear() noexcept {
        if (array) {
            cudaFree(array);
            array = nullptr;
        }
    }
public:
    /** Deleted copy constructor/assignment.
    **/
    CUDArray(This const&) = delete;
    This& operator=(This const&) = delete;
    /** Move constructor/assignment.
     * @param instance Instance to move
     * @return Current instance
    **/
    CUDArray(This&& instance) noexcept: array{instance.array}, length{instance.length} {
        instance.array = nullptr;
    }
    This& operator=(This&& instance) noexcept {
        clear();
        array  = instance.array;
        length = instance.length;
        instance.array = nullptr;
        return *this;
    }
    /** Empty constructor.
    **/
    CUDArray() noexcept: array{nullptr} {}
    /** Raw array constructor.
     * @param data Raw array
     * @param size Array size
     * @param conv Conversion function (optional)
    **/
    template<class CPUElem> CUDArray(CPUElem const* data, size_t size) {
        build<CPUElem>(data, size);
    }
    template<class CPUElem, class Conv> CUDArray(CPUElem const* data, size_t size, Conv&& conv) {
        build<CPUElem, Conv>(data, size, ::std::forward<Conv>(conv));
    }
    /** Vector constructor.
     * @param vect Vector
     * @param conv Conversion function (optional)
    **/
    template<class CPUElem, class Allocator> CUDArray(::std::vector<CPUElem, Allocator> const& vect) {
        build<CPUElem>(vect.data(), vect.size());
    }
    template<class CPUElem, class Allocator, class Conv> CUDArray(::std::vector<CPUElem, Allocator> const& vect, Conv&& conv) {
        build<CPUElem, Conv>(vect.data(), vect.size(), ::std::forward<Conv>(conv));
    }
    /** Clear destructor.
    **/
    ~CUDArray() {
        clear();
    }
public:
    /** GPU element access.
     * @param i Index of the GPU element to access
     * @return Accessed element
    **/
    GPUElem const& operator[](size_t i) const noexcept {
        return array[i];
    }
public:
    /** GPU array data getter.
     * @return GPU data pointer
    **/
    GPUElem const* data() const noexcept {
        return array;
    }
    /** GPU array size getter.
     * @return Array size
    **/
    size_t size() const noexcept {
        return length;
    }
};

// -------------------------------------------------------------------------- //
// GPU dynamic fixed-size array

/** Simple 'free'-calling deleter class.
**/
class VLCUDAFree final {
public:
    /** Free-calling deleter.
     * @param ptr Pointer to the memory to free
    **/
    void operator()(void* ptr) const noexcept {
        ::cudaFree(ptr);
    }
};

/** Variable-length (as in "non-constexpr const size") CUDA array class alias.
 * @param Type Element class
**/
template<class Type> using VLCUDArray = ::std::unique_ptr<Type[], VLCUDAFree>;

/** Allocate a "variable" (as in "non-constexpr const size") length array of the given size.
 * @param Type Element class
 * @param size Number of elements
 * @return Pointer on the first element of the array
**/
template<class Type> static VLCUDArray<Type> vlcudarray(size_t size) {
    Type* alloc;
    auto res = ::cudaMalloc(&alloc, size * sizeof(Type)); // Aligned on highest possible alignment by default
    if (unlikely(res != cudaSuccess))
        AT_ERROR("Unable to allocate memory on the GPU: ", cudaGetErrorString(res));
    return VLCUDArray<Type>{reinterpret_cast<typename VLCUDArray<Type>::pointer>(alloc)}; // Constructors are all marked noexcept
}
