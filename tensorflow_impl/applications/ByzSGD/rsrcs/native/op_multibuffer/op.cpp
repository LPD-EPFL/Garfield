/**
 * @file   op.cpp
 * @author Sébastien Rouault <sebastien.rouault@epfl.ch>
 *
 * @section LICENSE
 *
 * Copyright © 2019 Sébastien ROUAULT.
 *
 * @section DESCRIPTION
 *
 * MRMW read-only/write-only atomic register with optional "blocking producer-consumer" semantic, specialized for 'Buffer', TensorFlow custom operations.
**/

#include <cstring>
#include <memory>
#include <type_traits>
#include <utility>

#include <tensorflow/core/framework/allocator.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>

#include <common.hpp>
#include <buffer.hpp>
#include <multiregister.hpp>

using namespace tensorflow;
using namespace tensorflow::shape_inference;

using CPUDevice = Eigen::ThreadPoolDevice;

// -------------------------------------------------------------------------- //
// MultiRegister helpers

/** Specialization of MultiRegister class instance.
**/
using MultiBuffer = MultiRegister::MultiRegister<Buffer>;

/** Get an instance of a 'MultiBuffer' from its address.
 * @param addr Address of the register
 * @return Pointer to the instance of the register
**/
static MultiBuffer* from_address(int64 addr) noexcept {
    return reinterpret_cast<MultiBuffer*>(addr); // Blinking unsafe code
}
static_assert(sizeof(void*) <= sizeof(int64), "TensorFlow's 'int64' is not large enough to hold a pointer");

// -------------------------------------------------------------------------- //
// Op declarations and shape inferences

REGISTER_OP("MultibufferInput")
    .Output("output: dtype")
    .Attr("dtype: {float, double}")
    .Attr("register: int")
    .Attr("shape: shape")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* context) {
        DataType t;
        TF_RETURN_IF_ERROR(context->GetAttr("dtype", &t));
        TensorShape ts;
        TF_RETURN_IF_ERROR(context->GetAttr("shape", &ts));
        ShapeHandle s;
        TF_RETURN_IF_ERROR(context->MakeShapeFromTensorShape(ts, &s));
        context->set_output_handle_shapes_and_types(0, ::std::vector<shape_inference::ShapeAndType>{{s, t}});
        return Status::OK();
    });

REGISTER_OP("MultibufferOutput")
    .Input("input: dtype")
    .Attr("dtype: {float, double}")
    .Attr("register: int")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* context [[gnu::unused]]) {
        return Status::OK();
    });

// -------------------------------------------------------------------------- //
// Kernel implementations

/** Kernel implementation class template.
 * @param DType Tensor data type
**/
template<class DType> class Input: public OpKernel {
protected:
    MultiBuffer* instance; // Bound 'MultiBuffer' instance
    TensorShape  shape;    // Output tensor shape
public:
    explicit Input(OpKernelConstruction* context): OpKernel{context} {
        int64 addr;
        OP_REQUIRES_OK(context, context->GetAttr("register", &addr));
        instance = from_address(addr);
        OP_REQUIRES_OK(context, context->GetAttr("shape", &shape));
    }
public:
    void Compute(OpKernelContext* context) override {
        // Allocate output tensor
        Tensor* output_tn = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, shape, &output_tn));
        auto const size = sizeof(DType) * shape.num_elements(); // Size to copy, in bytes
        auto output = output_tn->flat<DType>().data();
        // Read buffer into the tensor
        try {
            auto read = instance->read();
            auto const* header = read.get().get_header(); // Get a pointer to the header of the buffer
            // Assertions
            if (unlikely(!header)) {
                context->SetStatus(errors::Internal("Given buffer was empty"));
                return;
            }
            if (unlikely(reinterpret_cast<uintptr_t>(output) % header->align != 0)) {
                context->SetStatus(errors::Internal("Given buffer invalid alignment, got ", output, " expected aligned on ", header->align));
                return;
            }
            if (unlikely(header->size < size)) {
                context->SetStatus(errors::Internal("Given buffer is too small, got ", header->size, ", expected (at least) ", size));
                return;
            }
            // Copy
            ::std::memcpy(output, header->get_buffer(), size);
        } catch (Exception::MultiRegister const& err) {
            context->SetStatus(errors::Internal("Internal 'MultiBuffer' exception while reading: ", err.what()));
            return;
        }
    }
};

/** Kernel implementation class template.
 * @param DType Tensor data type
**/
template<class DType> class Output: public OpKernel {
protected:
    MultiBuffer* instance; // Bound 'MultiBuffer' instance
public:
    explicit Output(OpKernelConstruction* context): OpKernel{context} {
        int64 addr;
        OP_REQUIRES_OK(context, context->GetAttr("register", &addr));
        instance = from_address(addr);
    }
public:
    void Compute(OpKernelContext* context) override {
        // Get input tensor
        auto&& input_tn = context->input(0);
        auto input = input_tn.flat<DType>().data();
        auto const size = sizeof(DType) * input_tn.shape().num_elements(); // Size to copy, in bytes
        // Write tensor into the buffer (may allocate)
        try {
            auto write = instance->write();
            auto* header = write.peek().get_header(); // Get a pointer to the header of the buffer
            if (header && header->size >= size) { // Overwrite buffer
                header->align = alignof(DType);
                header->size = size;
                ::std::memcpy(header->get_buffer(), input, size);
            } else { // (Re)allocate buffer
                Buffer buffer{alignof(DType), size};
                ::std::memcpy(buffer.get_header()->get_buffer(), input, size);
                write.set(::std::move(buffer));
            }
            write.validate();
        } catch (Exception::MultiRegister const& err) {
            context->SetStatus(errors::Internal("Internal 'MultiBuffer' exception while writing: ", err.what()));
            return;
        }
    }
};

// -------------------------------------------------------------------------- //
// Kernel builder registrations

REGISTER_KERNEL_BUILDER(Name("MultibufferInput").Device(DEVICE_CPU).TypeConstraint<float>("dtype"), Input<float>)
REGISTER_KERNEL_BUILDER(Name("MultibufferInput").Device(DEVICE_CPU).TypeConstraint<double>("dtype"), Input<double>)

REGISTER_KERNEL_BUILDER(Name("MultibufferOutput").Device(DEVICE_CPU).TypeConstraint<float>("dtype"), Output<float>)
REGISTER_KERNEL_BUILDER(Name("MultibufferOutput").Device(DEVICE_CPU).TypeConstraint<double>("dtype"), Output<double>)
