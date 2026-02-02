#include "rope_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>
#include <cstdint>
#include <vector>

namespace llaisys::ops::cpu {

template <typename T, typename PosT>
void rope_impl(T *out, const T *in, const PosT *pos_ids, 
               size_t seq_len, size_t n_heads, size_t head_dim, float theta) {
    
    if (head_dim % 2 != 0) {
        throw std::invalid_argument("RoPE: head dimension must be even");
    }
    
    const size_t half_dim = head_dim / 2;

    for (size_t s = 0; s < seq_len; s++) {
        const float position = static_cast<float>(pos_ids[s]);
        
        // 性能优化：为当前位置预计算该位置所有维度共享的 cos/sin
        std::vector<float> cos_buf(half_dim);
        std::vector<float> sin_buf(half_dim);
        
        for (size_t i = 0; i < half_dim; i++) {
            // 频率计算公式：freq = pos / theta^(2i/d)
            float freq = position / std::pow(theta, 2.0f * static_cast<float>(i) / static_cast<float>(head_dim));
            cos_buf[i] = std::cos(freq);
            sin_buf[i] = std::sin(freq);
        }

        for (size_t h = 0; h < n_heads; h++) {
            const T *head_in = in + (s * n_heads + h) * head_dim;
            T *head_out = out + (s * n_heads + h) * head_dim;

            for (size_t i = 0; i < half_dim; i++) {
                // 核心逻辑修正：配对方式为 i 和 i + half_dim
                const float a = llaisys::utils::cast<float>(head_in[i]);
                const float b = llaisys::utils::cast<float>(head_in[i + half_dim]);

                const float c = cos_buf[i];
                const float s_val = sin_buf[i];

                // 应用旋转变换
                // a' = a*cos - b*sin
                // b' = b*cos + a*sin
                head_out[i] = llaisys::utils::cast<T>(a * c - b * s_val);
                head_out[i + half_dim] = llaisys::utils::cast<T>(b * c + a * s_val);
            }
        }
    }
}


void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, 
          llaisysDataType_t type, llaisysDataType_t pos_type,
          size_t seq_len, size_t n_heads, size_t head_dim, float theta) {
    // 处理位置索引类型
    if (pos_type == LLAISYS_DTYPE_I64) {
        const int64_t* p_ids = reinterpret_cast<const int64_t*>(pos_ids);
        switch (type) {
            case LLAISYS_DTYPE_F32:
                rope_impl<float, int64_t>(reinterpret_cast<float*>(out), reinterpret_cast<const float*>(in), p_ids, seq_len, n_heads, head_dim, theta);
                break;
            case LLAISYS_DTYPE_F16:
                rope_impl<llaisys::fp16_t, int64_t>(reinterpret_cast<llaisys::fp16_t*>(out), reinterpret_cast<const llaisys::fp16_t*>(in), p_ids, seq_len, n_heads, head_dim, theta);
                break;
            case LLAISYS_DTYPE_BF16:
                rope_impl<llaisys::bf16_t, int64_t>(reinterpret_cast<llaisys::bf16_t*>(out), reinterpret_cast<const llaisys::bf16_t*>(in), p_ids, seq_len, n_heads, head_dim, theta);
                break;
            default:
                EXCEPTION_UNSUPPORTED_DATATYPE(type);
        }
    } else {
        // ... 处理 I32 情况 ...
        const int32_t* p_ids = reinterpret_cast<const int32_t*>(pos_ids);
        switch (type) {
            case LLAISYS_DTYPE_F32:
                rope_impl<float, int32_t>(reinterpret_cast<float*>(out), reinterpret_cast<const float*>(in), p_ids, seq_len, n_heads, head_dim, theta);
                break;
            case LLAISYS_DTYPE_F16:
                rope_impl<llaisys::fp16_t, int32_t>(reinterpret_cast<llaisys::fp16_t*>(out), reinterpret_cast<const llaisys::fp16_t*>(in), p_ids, seq_len, n_heads, head_dim, theta);
                break;
            case LLAISYS_DTYPE_BF16:
                rope_impl<llaisys::bf16_t, int32_t>(reinterpret_cast<llaisys::bf16_t*>(out), reinterpret_cast<const llaisys::bf16_t*>(in), p_ids, seq_len, n_heads, head_dim, theta);
                break;
            default:
                EXCEPTION_UNSUPPORTED_DATATYPE(type);
        }
    }
    

}
} // namespace llaisys::ops::cpu
