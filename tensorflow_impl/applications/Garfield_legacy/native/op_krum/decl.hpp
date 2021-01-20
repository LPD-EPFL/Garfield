/**
 * @file   decl.hpp
 * @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
 *
 * @section LICENSE
 *
 * Copyright © 2018-2019 École Polytechnique Fédérale de Lausanne (EPFL).
 * See LICENSE file.
 *
 * @section DESCRIPTION
 *
 * Multi-Krum GAR, declarations.
 *
 * Based on the algorithm introduced in the following paper:
 *   Blanchard Peva, El Mhamdi El Mahdi, Guerraoui Rachid, and Stainer Julien.
 *   Machine learning with adversaries: Byzantine tolerant gradient descent.
 *   In Advances in Neural Information Processing Systems 30, pp.118–128.
 *   Curran Associates, Inc., 2017.
**/

#pragma once

#include <type_traits>

#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>

#include <common.hpp>

using namespace tensorflow;
using namespace tensorflow::shape_inference;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

// -------------------------------------------------------------------------- //
// Op configuration

#define OP_NAME Krum
#define OP_TEXT TO_STRING(OP_NAME)

// -------------------------------------------------------------------------- //
// Helper classes

/** Equivalent TF-compatible type for 'size_t'.
**/
using tf_size_t = typename ::std::conditional<
    sizeof(size_t) == sizeof(uint32) && alignof(size_t) == alignof(uint32), uint32, typename ::std::conditional<
    sizeof(size_t) == sizeof(uint64) && alignof(size_t) == alignof(uint64), uint64, void>::type>::type; // Else unsupported target machine, 'void' to trigger a diagnostic

// -------------------------------------------------------------------------- //
// Kernel declaration
namespace OP_NAME {

template<class Device, class T> class Kernel: public Static {
public:
    static void process(OpKernelContext&, size_t const, size_t const, size_t const, size_t const, Tensor const&, Tensor&);
};

}
