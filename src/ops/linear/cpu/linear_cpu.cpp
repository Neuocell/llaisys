#include "linear_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>

namespace llaisys::ops::cpu {

// 内部通用的计算实现
template <typename T, typename AccT = float>
void linear_impl(T *out, const T *in, const T *weight, const T *bias,
                size_t batch_size, size_t in_features, size_t out_features, bool has_bias) {
    
    for (size_t b = 0; b < batch_size; b++) {
        const T *batch_in = in + b * in_features;
        T *batch_out = out + b * out_features;

        for (size_t o = 0; o < out_features; o++) {
            // 使用 AccT (通常为 float) 进行高精度累加，减少舍入误差(原来反复转换类型会增加误差)
            AccT sum = has_bias ? llaisys::utils::cast<AccT>(bias[o]) : static_cast<AccT>(0.0f);
            
            const T *weight_row = weight + o * in_features;

            // 内循环：计算点积。此处 weight_row 的访问是连续的，Cache 友好
            for (size_t i = 0; i < in_features; i++) {
                AccT in_val = llaisys::utils::cast<AccT>(batch_in[i]);
                AccT w_val = llaisys::utils::cast<AccT>(weight_row[i]);
                sum += in_val * w_val;
            }

            // 最终结果转回目标类型 T
            batch_out[o] = llaisys::utils::cast<T>(sum);
        }
    }
}

void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t type, size_t batch_size, size_t in_features, size_t out_features,
            bool has_bias) {
    
    switch (type) {
    case LLAISYS_DTYPE_F32:
        linear_impl<float, float>(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(in),
            reinterpret_cast<const float *>(weight),
            reinterpret_cast<const float *>(bias),
            batch_size, in_features, out_features, has_bias);
        break;
    case LLAISYS_DTYPE_BF16:
        linear_impl<llaisys::bf16_t, float>(
            reinterpret_cast<llaisys::bf16_t *>(out),
            reinterpret_cast<const llaisys::bf16_t *>(in),
            reinterpret_cast<const llaisys::bf16_t *>(weight),
            reinterpret_cast<const llaisys::bf16_t *>(bias),
            batch_size, in_features, out_features, has_bias);
        break;
    case LLAISYS_DTYPE_F16:
        linear_impl<llaisys::fp16_t, float>(
            reinterpret_cast<llaisys::fp16_t *>(out),
            reinterpret_cast<const llaisys::fp16_t *>(in),
            reinterpret_cast<const llaisys::fp16_t *>(weight),
            reinterpret_cast<const llaisys::fp16_t *>(bias),
            batch_size, in_features, out_features, has_bias);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::cpu