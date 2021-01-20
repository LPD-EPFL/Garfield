/**
 * @file   op.cpp
 * @author Sébastien Rouault <sebastien.rouault@epfl.ch>
 *
 * @section LICENSE
 *
 * Copyright © 2018-2019 Sébastien ROUAULT.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * @section DESCRIPTION
 *
 * Median GAR, TensorFlow custom operation.
**/

#include <common.hpp>
#include "decl.hpp"

// -------------------------------------------------------------------------- //
// Op declaration and shape inference

REGISTER_OP(OP_TEXT)
    .Input("gradients: T")
    .Output("aggregated: T")
    .Attr("T: {float, double}")
    .SetShapeFn([](InferenceContext* c) {
        auto&& input_tn = c->input(0);
        ShapeHandle dummy;
        TF_RETURN_IF_ERROR(c->WithRank(input_tn, 2, &dummy));
        c->set_output(0, c->MakeShape(::std::vector<DimensionHandle>{c->Dim(input_tn, 1)}));
        return Status::OK();
    });

// -------------------------------------------------------------------------- //
// Interface implementation
namespace OP_NAME {

template<class Device, class T> class Interface: public OpKernel {
public:
    explicit Interface(OpKernelConstruction* context): OpKernel{context} {}
public:
    void Compute(OpKernelContext* context) override {
        Tensor const& input_tn = context->input(0);
        OP_REQUIRES(context, input_tn.NumElements() <= tensorflow::kint32max, errors::InvalidArgument("Too many elements in tensor"));
        auto n = input_tn.dim_size(0);
        auto d = input_tn.dim_size(1);
        Tensor* output_tn = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{d}, &output_tn));
        Kernel<Device, T>::process(*context, n, d, input_tn, *output_tn);
    }
};

}
// -------------------------------------------------------------------------- //
// Interface-kernel registrations

// CPU kernel
#define REGISTER_CPU(T) \
    extern template class OP_NAME::Kernel<CPUDevice, T>; \
    REGISTER_KERNEL_BUILDER(Name(OP_TEXT).Device(DEVICE_CPU).TypeConstraint<T>("T"), OP_NAME::Interface<CPUDevice, T>)
REGISTER_CPU(float);
REGISTER_CPU(double);
#undef REGISTER_CPU

// GPU kernel
#ifdef GOOGLE_CUDA

#define REGISTER_GPU(T) \
    extern template class OP_NAME::Kernel<GPUDevice, T>; \
    REGISTER_KERNEL_BUILDER(Name(OP_TEXT).Device(DEVICE_GPU).TypeConstraint<T>("T"), OP_NAME::Interface<GPUDevice, T>)
REGISTER_GPU(float);
REGISTER_GPU(double);
#undef REGISTER_GPU

#endif
