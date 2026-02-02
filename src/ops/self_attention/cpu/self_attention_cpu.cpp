#include "self_attention_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>
#include <algorithm>
#include <limits>
#include <vector>

namespace llaisys::ops::cpu {

/**
 * 高性能 Softmax 实现（针对 float 数组优化）
 */
void softmax_inplace(float* x, size_t size) {
    float max_val = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < size; ++i) max_val = std::max(max_val, x[i]);

    float sum = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        x[i] = std::exp(x[i] - max_val);
        sum += x[i];
    }


//在现代 CPU 上，除法指令比乘法指令慢得多
// （通常需要 10-20 个时钟周期，而乘法只需 3-5 个）。
// 在循环中重复除法会显著降低性能。
    float inv_sum = 1.0f / sum;
    for (size_t i = 0; i < size; ++i) x[i] *= inv_sum;
}

template <typename T>
void self_attention_impl(T *attn_val, const T *q, const T *k, const T *v,
                        size_t q_len, size_t kv_len, size_t n_heads, size_t n_kv_heads, 
                        size_t head_dim, float scale) {
    //GQA/MQA 支持 确保多个 Q 头可以共享同一组 KV 头。
    const size_t head_repeats = n_heads / n_kv_heads;
    // past_len 指的是 KV Cache 中已有的长度
    const size_t past_len = kv_len - q_len; 

    // 预分配缓冲区，避免在循环内反复动态分配
    // 注意：在多线程环境下，每个线程需要独立的缓冲区
    std::vector<float> score_buf(kv_len);
    // qi: 当前正在处理 Query 序列中的第几个位置
    for (size_t qi = 0; qi < q_len; qi++) {
        // current_q_pos: 该词在整段对话中的全局位置（考虑了 KV Cache 的历史长度）
        size_t current_q_pos = past_len + qi; 
        // h: 遍历每一个注意力头
        for (size_t h = 0; h < n_heads; h++) {
            // kv_h: 计算该 Q 头对应的 K/V 头索引 (支持 GQA/MQA 架构)
            size_t kv_h = h / head_repeats;
            // q_ptr: 定位到当前 Q 的向量起点
            const T *q_ptr = q + (qi * n_heads + h) * head_dim;

            // 1. 计算 Q * K^T 并应用 Scale 和 Causal Mask
            for (size_t kj = 0; kj < kv_len; kj++) {
                // 因果掩码：如果 Key 的位置 kj 大于当前 Query 的位置，
                // 说明这是“未来”，必须遮住
                if (kj > current_q_pos) {
                    //使用 -10000.0f 代替 -inf。在某些编译器优化下，
                    // -inf 可能导致特定的浮点异常，
                    // 而大负数足以让 Softmax 的权重趋于 0。
                    score_buf[kj] = -10000.0f; // Causal Mask
                    continue;
                }
                // k_ptr: 定位到要匹配的 Key
                const T *k_ptr = k + (kj * n_kv_heads + kv_h) * head_dim;
                float dot = 0.0f;
                for (size_t d = 0; d < head_dim; d++) {
                    dot += llaisys::utils::cast<float>(q_ptr[d]) * llaisys::utils::cast<float>(k_ptr[d]);
                }
                // 存入临时缓冲区并应用缩放
                score_buf[kj] = dot * scale;
            }

            // 2. Softmax
            softmax_inplace(score_buf.data(), kv_len);

            // 3. 计算加权和 (Attention Scores * V)
            T *out_ptr = attn_val + (qi * n_heads + h) * head_dim;
            
            // 为了精度和速度，使用 float 累加器
            for (size_t d = 0; d < head_dim; d++) {
                float sum = 0.0f;
                // 只遍历到 current_q_pos，因为后面的权重被 Mask 掉后全是 0
                for (size_t kj = 0; kj <= current_q_pos; kj++) {
                    const T *v_ptr = v + (kj * n_kv_heads + kv_h) * head_dim;
                    sum += score_buf[kj] * llaisys::utils::cast<float>(v_ptr[d]);
                }
                out_ptr[d] = llaisys::utils::cast<T>(sum);
            }
        }
    }
}

void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v,
                    llaisysDataType_t type, size_t q_len, size_t kv_len, 
                    size_t n_heads, size_t n_kv_heads, size_t head_dim, float scale) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return self_attention_impl(reinterpret_cast<float *>(attn_val), 
                              reinterpret_cast<const float *>(q),
                              reinterpret_cast<const float *>(k),
                              reinterpret_cast<const float *>(v),
                              q_len, kv_len, n_heads, n_kv_heads, head_dim, scale);
    case LLAISYS_DTYPE_BF16:
        return self_attention_impl(reinterpret_cast<llaisys::bf16_t *>(attn_val), 
                              reinterpret_cast<const llaisys::bf16_t *>(q),
                              reinterpret_cast<const llaisys::bf16_t *>(k),
                              reinterpret_cast<const llaisys::bf16_t *>(v),
                              q_len, kv_len, n_heads, n_kv_heads, head_dim, scale);
    case LLAISYS_DTYPE_F16:
        return self_attention_impl(reinterpret_cast<llaisys::fp16_t *>(attn_val), 
                              reinterpret_cast<const llaisys::fp16_t *>(q),
                              reinterpret_cast<const llaisys::fp16_t *>(k),
                              reinterpret_cast<const llaisys::fp16_t *>(v),
                              q_len, kv_len, n_heads, n_kv_heads, head_dim, scale);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    
    }
}

} // namespace llaisys::ops::cpu