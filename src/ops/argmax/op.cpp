
#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/argmax_cpu.hpp"

namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    // 参数检查
    CHECK_SAME_DEVICE(max_idx, max_val, vals);
    ASSERT(max_idx->ndim() == 1 && max_idx->shape()[0] == 1, 
           "argmax: max_idx must be scalar tensor with shape (1,)");
    ASSERT(max_val->ndim() == 1 && max_val->shape()[0] == 1, 
           "argmax: max_val must be scalar tensor with shape (1,)");
    ASSERT(vals->ndim() >= 1, "argmax: input tensor must have at least 1 dimension");
    
    // 检查数据类型
    CHECK_SAME_DTYPE(max_val->dtype(), vals->dtype());
    ASSERT(max_idx->dtype() == LLAISYS_DTYPE_I64, 
           "argmax: max_idx must have int64 data type");
    
    // 只支持连续张量
    ASSERT(vals->isContiguous(), "argmax: input tensor must be contiguous");
    
    // 计算总元素数
    size_t numel = vals->numel();
    
    // 只支持CPU计算
    if (vals->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::argmax(max_idx->data(), max_val->data(), vals->data(), 
                          vals->dtype(), numel);
    }
    
    llaisys::core::context().setDevice(vals->deviceType(), vals->deviceId());
    
    switch (vals->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::argmax(max_idx->data(), max_val->data(), vals->data(), 
                          vals->dtype(), numel);
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
