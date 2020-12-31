/**
 * @file   pytorch.hpp
 * @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
 *
 * @section LICENSE
 *
 * Copyright © 2018-2019 École Polytechnique Fédérale de Lausanne (EPFL).
 * All rights reserved.
 *
 * @section DESCRIPTION
 *
 * Imports for PyTorch extensions.
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
#include <torch/extension.h>
#include <torch/script.h>

// -------------------------------------------------------------------------- //
// Error assertion helper

// Main helper
#ifndef NDEBUG
    #define CUDA_DO_ASSERT(expr, file, line) \
    { \
        auto _code = (expr); \
        if (unlikely(_code != cudaSuccess)) \
            AT_ERROR("CUDA error: ", cudaGetErrorString(_code), " at ", file, ":", line); \
    }
#else
    #define CUDA_DO_ASSERT(expr, file, line)
#endif

// Convenience wrappers
#define CUDA_ASSERT(expr) \
    CUDA_DO_ASSERT(expr, __FILE__, __LINE__)
#define CUDA_ASSERT_KERNEL() \
    CUDA_DO_ASSERT(cudaPeekAtLastError(), __FILE__, __LINE__); \
    CUDA_DO_ASSERT(cudaDeviceSynchronize(), __FILE__, __LINE__)
