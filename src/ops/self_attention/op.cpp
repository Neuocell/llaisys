#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/self_attention_cpu.hpp"

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());
    
    // 检查张量形状
    // q 形状: [q_len, n_heads, head_dim]
    // k 形状: [kv_len, n_kv_heads, head_dim]
    // v 形状: [kv_len, n_kv_heads, head_dim]
    // attn_val 形状: [q_len, n_heads, head_dim]
    ASSERT(q->ndim() == 3, "self_attention: query tensor must have 3 dimensions");
    ASSERT(k->ndim() == 3, "self_attention: key tensor must have 3 dimensions");
    ASSERT(v->ndim() == 3, "self_attention: value tensor must have 3 dimensions");
    ASSERT(attn_val->ndim() == 3, "self_attention: output tensor must have 3 dimensions");
    
    size_t q_len = q->shape()[0];
    size_t n_heads = q->shape()[1];
    size_t head_dim = q->shape()[2];
    
    size_t kv_len = k->shape()[0];
    size_t n_kv_heads = k->shape()[1];
    
    ASSERT(v->shape()[0] == kv_len && v->shape()[1] == n_kv_heads && v->shape()[2] == head_dim,
           "self_attention: value tensor shape must match key tensor shape");
    ASSERT(attn_val->shape()[0] == q_len && attn_val->shape()[1] == n_heads && attn_val->shape()[2] == head_dim,
           "self_attention: output tensor shape must match query tensor shape");
    
    // 检查多头配置
    ASSERT(n_heads % n_kv_heads == 0, 
           "self_attention: number of query heads must be divisible by number of key/value heads");
    
    // 检查 scale 参数
    ASSERT(scale > 0.0f, "self_attention: scale must be positive");
    
    // 检查张量是否连续
    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(),
           "self_attention: all tensors must be contiguous");

    // 总是支持 CPU 计算
    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(),
                                  attn_val->dtype(), q_len, kv_len, n_heads, n_kv_heads, head_dim, scale);
    }

    llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());

    switch (attn_val->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(),
                                  attn_val->dtype(), q_len, kv_len, n_heads, n_kv_heads, head_dim, scale);
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
