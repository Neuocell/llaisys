#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/linear_cpu.hpp"

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight);
    
    // 检查偏置张量的设备（如果存在）
    if (bias != nullptr) {
        CHECK_SAME_DEVICE(out, bias);
    }
    
    // 检查数据类型匹配
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    if (bias != nullptr) {
        CHECK_SAME_DTYPE(out->dtype(), bias->dtype());
    }
    
    // 检查张量形状
    // in 形状: [batch_size, in_features]
    // weight 形状: [out_features, in_features]
    // bias 形状: [out_features] 或 nullptr
    // out 形状: [batch_size, out_features]
    ASSERT(in->ndim() >= 2, "linear: input tensor must have at least 2 dimensions");
    ASSERT(weight->ndim() == 2, "linear: weight tensor must have 2 dimensions");
    ASSERT(out->ndim() >= 2, "linear: output tensor must have at least 2 dimensions");
    
    size_t batch_size = 1;
    for (size_t i = 0; i < in->ndim() - 1; i++) {
        batch_size *= in->shape()[i];
    }
    size_t in_features = in->shape()[in->ndim() - 1];
    size_t out_features = weight->shape()[0];
    
    ASSERT(weight->shape()[1] == in_features,
           "linear: weight second dimension must match input last dimension");
    ASSERT(out->shape()[out->ndim() - 1] == out_features,
           "linear: output last dimension must match weight first dimension");
    
    // 检查批次维度是否匹配
    size_t out_batch_size = 1;
    for (size_t i = 0; i < out->ndim() - 1; i++) {
        out_batch_size *= out->shape()[i];
    }
    ASSERT(out_batch_size == batch_size,
           "linear: output batch size must match input batch size");
    
    // 检查偏置形状
    bool has_bias = (bias != nullptr);
    if (has_bias) {
        ASSERT(bias->ndim() == 1, "linear: bias tensor must have 1 dimension");
        ASSERT(bias->shape()[0] == out_features,
               "linear: bias dimension must match output features");
    }
    
    // 检查张量是否连续
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),
           "linear: output, input and weight tensors must be contiguous");
    if (has_bias) {
        ASSERT(bias->isContiguous(), "linear: bias tensor must be contiguous");
    }

    // 总是支持 CPU 计算
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::linear(out->data(), in->data(), weight->data(), 
                          has_bias ? bias->data() : nullptr,
                          out->dtype(), batch_size, in_features, out_features, has_bias);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(out->data(), in->data(), weight->data(), 
                          has_bias ? bias->data() : nullptr,
                          out->dtype(), batch_size, in_features, out_features, has_bias);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
