#include "rope_cpu.hpp"

#include "../../../utils.hpp"
#include <cmath>
#include <cstdint>

// RoPE 核心计算模板
template <typename T, typename PosT>
void rope_(T *out, const T *in, const PosT *pos_ids, 
           size_t seq_len, size_t n_heads, size_t head_dim, float theta) {
    // 检查头维度是否为偶数
    if (head_dim % 2 != 0) {
        throw std::invalid_argument("RoPE: head dimension must be even");
    }
    
    size_t half_dim = head_dim / 2;
    
    for (size_t s = 0; s < seq_len; s++) {
        float position = static_cast<float>(pos_ids[s]);
        
        for (size_t h = 0; h < n_heads; h++) {
            const T *head_in = in + (s * n_heads + h) * head_dim;
            T *head_out = out + (s * n_heads + h) * head_dim;
            
            // 计算旋转频率
            for (size_t i = 0; i < half_dim; i++) {
                float freq = position / std::pow(theta, 2.0f * static_cast<float>(i) / static_cast<float>(head_dim));
                float sin_val = std::sin(freq);
                float cos_val = std::cos(freq);
                
                // 获取输入的前半部分和后半部分
                T a, b;
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    a = head_in[i];
                    b = head_in[i + half_dim];
                    
                    float a_f = llaisys::utils::cast<float>(a);
                    float b_f = llaisys::utils::cast<float>(b);
                    
                    // 应用旋转：a' = a * cos - b * sin, b' = b * cos + a * sin
                    float a_out = a_f * cos_val - b_f * sin_val;
                    float b_out = b_f * cos_val + a_f * sin_val;
                    
                    head_out[i] = llaisys::utils::cast<T>(a_out);
                    head_out[i + half_dim] = llaisys::utils::cast<T>(b_out);
                } else {
                    a = head_in[i];
                    b = head_in[i + half_dim];
                    
                    // 应用旋转：a' = a * cos - b * sin, b' = b * cos + a * sin
                    head_out[i] = static_cast<T>(a * cos_val - b * sin_val);
                    head_out[i + half_dim] = static_cast<T>(b * cos_val + a * sin_val);
                }
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, 
          llaisysDataType_t type, llaisysDataType_t pos_type,
          size_t seq_len, size_t n_heads, size_t head_dim, float theta) {
    // 处理位置索引类型
    switch (pos_type) {
    case LLAISYS_DTYPE_I32:
        // 处理数据类型
        switch (type) {
        case LLAISYS_DTYPE_F32:
            return rope_(reinterpret_cast<float *>(out), 
                        reinterpret_cast<const float *>(in),
                        reinterpret_cast<const int32_t *>(pos_ids),
                        seq_len, n_heads, head_dim, theta);
        case LLAISYS_DTYPE_BF16:
            return rope_(reinterpret_cast<llaisys::bf16_t *>(out), 
                        reinterpret_cast<const llaisys::bf16_t *>(in),
                        reinterpret_cast<const int32_t *>(pos_ids),
                        seq_len, n_heads, head_dim, theta);
        case LLAISYS_DTYPE_F16:
            return rope_(reinterpret_cast<llaisys::fp16_t *>(out), 
                        reinterpret_cast<const llaisys::fp16_t *>(in),
                        reinterpret_cast<const int32_t *>(pos_ids),
                        seq_len, n_heads, head_dim, theta);
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(type);
        }
        break;
    case LLAISYS_DTYPE_I64:
        // 处理数据类型
        switch (type) {
        case LLAISYS_DTYPE_F32:
            return rope_(reinterpret_cast<float *>(out), 
                        reinterpret_cast<const float *>(in),
                        reinterpret_cast<const int64_t *>(pos_ids),
                        seq_len, n_heads, head_dim, theta);
        case LLAISYS_DTYPE_BF16:
            return rope_(reinterpret_cast<llaisys::bf16_t *>(out), 
                        reinterpret_cast<const llaisys::bf16_t *>(in),
                        reinterpret_cast<const int64_t *>(pos_ids),
                        seq_len, n_heads, head_dim, theta);
        case LLAISYS_DTYPE_F16:
            return rope_(reinterpret_cast<llaisys::fp16_t *>(out), 
                        reinterpret_cast<const llaisys::fp16_t *>(in),
                        reinterpret_cast<const int64_t *>(pos_ids),
                        seq_len, n_heads, head_dim, theta);
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(type);
        }
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(pos_type);
    }
}
} // namespace llaisys::ops::cpu
