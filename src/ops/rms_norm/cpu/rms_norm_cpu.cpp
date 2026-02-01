#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"
#include <cmath>
#include <algorithm>

// RMSNorm 核心计算模板
template <typename T>
void rms_norm_(T *out, const T *in, const T *weight, size_t rows, size_t cols, float eps) {
    for (size_t i = 0; i < rows; i++) {
        const T *row_in = in + i * cols;
        T *row_out = out + i * cols;
        
        // 计算均方根
        float sum_sq = 0.0f;
        for (size_t j = 0; j < cols; j++) {
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                float val = llaisys::utils::cast<float>(row_in[j]);
                sum_sq += val * val;
            } else {
                float val = static_cast<float>(row_in[j]);
                sum_sq += val * val;
            }
        }
        
        float mean_sq = sum_sq / static_cast<float>(cols);
        float rms = 1.0f / std::sqrt(mean_sq + eps);
        
        // 应用归一化和权重
        for (size_t j = 0; j < cols; j++) {
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                float in_val = llaisys::utils::cast<float>(row_in[j]);
                float weight_val = llaisys::utils::cast<float>(weight[j]);
                float result = in_val * rms * weight_val;
                row_out[j] = llaisys::utils::cast<T>(result);
            } else {
                row_out[j] = static_cast<T>(row_in[j] * rms * weight[j]);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, 
              llaisysDataType_t type, size_t rows, size_t cols, float eps) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_(reinterpret_cast<float *>(out), 
                        reinterpret_cast<const float *>(in),
                        reinterpret_cast<const float *>(weight), 
                        rows, cols, eps);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_(reinterpret_cast<llaisys::bf16_t *>(out), 
                        reinterpret_cast<const llaisys::bf16_t *>(in),
                        reinterpret_cast<const llaisys::bf16_t *>(weight), 
                        rows, cols, eps);
    case LLAISYS_DTYPE_F16:
        return rms_norm_(reinterpret_cast<llaisys::fp16_t *>(out), 
                        reinterpret_cast<const llaisys::fp16_t *>(in),
                        reinterpret_cast<const llaisys::fp16_t *>(weight), 
                        rows, cols, eps);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
