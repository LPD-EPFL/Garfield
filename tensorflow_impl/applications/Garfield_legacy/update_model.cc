#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"


using namespace tensorflow;

REGISTER_OP("UpdateModel")
    .Input("model: float32")
    .Input("grad: float32")
    .Output("updated_model: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });


class UpdateModel : public OpKernel {
 public:
  explicit UpdateModel(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto model = input_tensor.flat<float>();

    const Tensor& grad_tensor = context->input(1);
    auto grad = grad_tensor.flat<float>();


    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<float>();

    // apply the gradient to the model.
    const int N = model.size();
    for (int i = 0; i < N; i++) {
      output_flat(i) = model(i) - 0.1*grad(i);
    }

  }
};

REGISTER_KERNEL_BUILDER(Name("UpdateModel").Device(DEVICE_CPU), UpdateModel);

