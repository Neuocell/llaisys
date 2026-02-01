#include "swiglu_cpu.hpp"

#include "../../../utils.hpp"
#include <cmath>

// Sigmoid 函数模板实现
template <typename T>
T sigmoid_(T x) {
    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
        // 对于半精度类型，先转换为 float 计算，再转换回来
        float x_f = llaisys::utils::cast<float>(x);
        float result = 1.0f / (1.0f + std::exp(-x_f));
        return llaisys::utils::cast<T>(result);
    } else {
        // 对于单精度和双精度，直接计算
        return static_cast<T>(1.0) / (static_cast<T>(1.0) + std::exp(-x));
    }
}

// SwiGLU 核心计算模板
template <typename T>
void swiglu_(T *out, const T *gate, const T *up, size_t numel) {
    for (size_t i = 0; i < numel; i++) {
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            // 半精度类型需要特殊处理
            float gate_f = llaisys::utils::cast<float>(gate[i]);
            float up_f = llaisys::utils::cast<float>(up[i]);
            
            // 计算 SwiGLU: out = up * (gate * sigmoid(gate))
            float sigmoid_gate = 1.0f / (1.0f + std::exp(-gate_f));
            float result = up_f * (gate_f * sigmoid_gate);
            
            out[i] = llaisys::utils::cast<T>(result);
        } else {
            // 全精度类型直接计算
            T sigmoid_gate = static_cast<T>(1.0) / (static_cast<T>(1.0) + std::exp(-gate[i]));
            out[i] = up[i] * (gate[i] * sigmoid_gate);
        }
    }
}

namespace llaisys::ops::cpu {
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up, llaisysDataType_t type, size_t numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return swiglu_(reinterpret_cast<float *>(out), 
                      reinterpret_cast<const float *>(gate), 
                      reinterpret_cast<const float *>(up), numel);
    case LLAISYS_DTYPE_BF16:
        return swiglu_(reinterpret_cast<llaisys::bf16_t *>(out), 
                      reinterpret_cast<const llaisys::bf16_t *>(gate), 
                      reinterpret_cast<const llaisys::bf16_t *>(up), numel);
    case LLAISYS_DTYPE_F16:
        return swiglu_(reinterpret_cast<llaisys::fp16_t *>(out), 
                      reinterpret_cast<const llaisys::fp16_t *>(gate), 
                      reinterpret_cast<const llaisys::fp16_t *>(up), numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
