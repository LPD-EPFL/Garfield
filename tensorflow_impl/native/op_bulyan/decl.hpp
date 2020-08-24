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
 * Bulyan over Multi-Krum GAR, declarations.
 *
 * Based on the algorithm introduced in the following paper:
 *   El Mhamdi El Mahdi, Guerraoui Rachid, and Rouault Sébastien.
 *   The Hidden Vulnerability of Distributed Learning in Byzantium.
 *   In Dy, J. and Krause, A. (eds.), Proceedings of the 35th International
 *   Conference on Machine Learning, volume 80 of Proceedings of Machine
 *   Learning  Research, pp. 3521-3530, Stockholmsmässan, Stockholm Sweden,
 *   10-15 Jul 2018. PMLR. URL http://proceedings.mlr.press/v80/mhamdi18a.html.
**/

#pragma once

#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>

#include <common.hpp>

using namespace tensorflow;
using namespace tensorflow::shape_inference;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

// -------------------------------------------------------------------------- //
// Op configuration

#define OP_NAME Bulyan
#define OP_TEXT TO_STRING(OP_NAME)

// -------------------------------------------------------------------------- //
// Kernel declaration
namespace OP_NAME {

template<class Device, class T> class Kernel: public Static {
public:
    static void process(OpKernelContext&, size_t const, size_t const, size_t const, size_t const, Tensor const&, Tensor&);
};

}
