#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/embedding_cpu.hpp"

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);
    
    // 检查索引张量的数据类型
    ASSERT(index->dtype() == LLAISYS_DTYPE_I32 || index->dtype() == LLAISYS_DTYPE_I64,
           "embedding: index tensor must have int32 or int64 data type");
    
    // 检查权重和输出张量的数据类型是否匹配
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());
    
    // 检查张量形状
    // index 形状: [batch_size] 或 [batch_size, seq_len]
    // weight 形状: [vocab_size, embedding_dim]
    // out 形状: [batch_size, embedding_dim] 或 [batch_size, seq_len, embedding_dim]
    ASSERT(weight->ndim() == 2, "embedding: weight tensor must have 2 dimensions");
    ASSERT(out->ndim() >= 2, "embedding: output tensor must have at least 2 dimensions");
    ASSERT(index->ndim() >= 1, "embedding: index tensor must have at least 1 dimension");
    
    
    size_t embedding_dim = weight->shape()[1];
    
    // 检查输出张量的最后一个维度是否匹配 embedding_dim
    ASSERT(out->shape()[out->ndim() - 1] == embedding_dim,
           "embedding: output tensor last dimension must match embedding dimension");
    
    // 计算索引张量的总元素数
    size_t index_size = index->numel();
    
    // 检查张量是否连续
    ASSERT(out->isContiguous() && index->isContiguous() && weight->isContiguous(),
           "embedding: all tensors must be contiguous");

    // 总是支持 CPU 计算
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::embedding(out->data(), index->data(), weight->data(), 
                             out->dtype(), index->dtype(), index_size, embedding_dim);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::embedding(out->data(), index->data(), weight->data(), 
                             out->dtype(), index->dtype(), index_size, embedding_dim);
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
