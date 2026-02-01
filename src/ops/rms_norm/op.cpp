#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rms_norm_cpu.hpp"

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out, in, weight);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    
    // 检查张量形状
    // in 形状: [batch_size, hidden_size]
    // weight 形状: [hidden_size]
    // out 形状: [batch_size, hidden_size]
    ASSERT(in->ndim() >= 2, "rms_norm: input tensor must have at least 2 dimensions");
    ASSERT(weight->ndim() == 1, "rms_norm: weight tensor must have 1 dimension");
    ASSERT(out->ndim() >= 2, "rms_norm: output tensor must have at least 2 dimensions");
    
    size_t hidden_size = in->shape()[in->ndim() - 1];
    ASSERT(weight->shape()[0] == hidden_size,
           "rms_norm: weight dimension must match input last dimension");
    ASSERT(out->shape()[out->ndim() - 1] == hidden_size,
           "rms_norm: output last dimension must match input last dimension");
    
    // 检查批次维度是否匹配
    size_t batch_size = 1;
    for (size_t i = 0; i < in->ndim() - 1; i++) {
        batch_size *= in->shape()[i];
    }
    ASSERT(out->numel() == batch_size * hidden_size,
           "rms_norm: output tensor size must match input tensor size");
    
    // 检查张量是否连续
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),
           "rms_norm: all tensors must be contiguous");
    
    // 检查 eps 参数
    ASSERT(eps > 0.0f, "rms_norm: eps must be positive");

    // 总是支持 CPU 计算
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rms_norm(out->data(), in->data(), weight->data(), 
                            out->dtype(), batch_size, hidden_size, eps);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rms_norm(out->data(), in->data(), weight->data(), 
                            out->dtype(), batch_size, hidden_size, eps);
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
