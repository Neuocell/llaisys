#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rope_cpu.hpp"

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);
    
    // 检查数据类型匹配
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    
    // 检查位置索引张量的数据类型
    ASSERT(pos_ids->dtype() == LLAISYS_DTYPE_I32 || pos_ids->dtype() == LLAISYS_DTYPE_I64,
           "rope: position indices tensor must have int32 or int64 data type");
    
    // 检查张量形状
    // in 形状: [seq_len, n_heads, head_dim]
    // pos_ids 形状: [seq_len]
    // out 形状: [seq_len, n_heads, head_dim]
    ASSERT(in->ndim() == 3, "rope: input tensor must have 3 dimensions");
    ASSERT(pos_ids->ndim() == 1, "rope: position indices tensor must have 1 dimension");
    ASSERT(out->ndim() == 3, "rope: output tensor must have 3 dimensions");
    
    size_t seq_len = in->shape()[0];
    size_t n_heads = in->shape()[1];
    size_t head_dim = in->shape()[2];
    
    ASSERT(pos_ids->shape()[0] == seq_len,
           "rope: position indices length must match sequence length");
    ASSERT(out->shape()[0] == seq_len && out->shape()[1] == n_heads && out->shape()[2] == head_dim,
           "rope: output shape must match input shape");
    
    // 检查头维度是否为偶数
    ASSERT(head_dim % 2 == 0, "rope: head dimension must be even");
    
    // 检查 theta 参数
    ASSERT(theta > 0.0f, "rope: theta must be positive");
    
    // 检查张量是否连续
    ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(),
           "rope: all tensors must be contiguous");

    // 总是支持 CPU 计算
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rope(out->data(), in->data(), pos_ids->data(), 
                        out->dtype(), pos_ids->dtype(),
                        seq_len, n_heads, head_dim, theta);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rope(out->data(), in->data(), pos_ids->data(), 
                        out->dtype(), pos_ids->dtype(),
                        seq_len, n_heads, head_dim, theta);
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

