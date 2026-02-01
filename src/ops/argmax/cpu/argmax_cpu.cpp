// /home/wangls/llaisys/src/ops/argmax/cpu/argmax_cpu.cpp
#include "argmax_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <limits>

namespace llaisys::ops::cpu {

template <typename T>
void argmax_(int64_t *max_idx, T *max_val, const T *vals, size_t numel) {
    if (numel == 0) {
        *max_idx = -1;
        // 修复：使用utils::cast进行类型安全的初始化
        *max_val = llaisys::utils::cast<T>(0.0f);
        return;
    }
    
    T current_max = vals[0];
    int64_t current_max_idx = 0;
    
    for (size_t i = 1; i < numel; i++) {
        // 处理半精度浮点数需要特殊处理
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            float current_val = llaisys::utils::cast<float>(vals[i]);
            float current_max_val = llaisys::utils::cast<float>(current_max);
            //比较的时候转为全精度，是因为没有支持的比较运算符重载？？
            if (current_val > current_max_val) {
                current_max = vals[i];
                current_max_idx = i;
            }
        } else {
            if (vals[i] > current_max) {
                current_max = vals[i];
                current_max_idx = i;
            }
        }
    }
    
    *max_idx = current_max_idx;
    *max_val = current_max;
}

void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, 
            llaisysDataType_t vals_type, size_t numel) {
    // max_idx 总是int64类型
    int64_t *max_idx_ptr = reinterpret_cast<int64_t *>(max_idx);
    
    switch (vals_type) {
    case LLAISYS_DTYPE_F32:
        return argmax_(max_idx_ptr, reinterpret_cast<float *>(max_val), 
                      reinterpret_cast<const float *>(vals), numel);
    case LLAISYS_DTYPE_BF16:
        return argmax_(max_idx_ptr, reinterpret_cast<llaisys::bf16_t *>(max_val), 
                      reinterpret_cast<const llaisys::bf16_t *>(vals), numel);
    case LLAISYS_DTYPE_F16:
        return argmax_(max_idx_ptr, reinterpret_cast<llaisys::fp16_t *>(max_val), 
                      reinterpret_cast<const llaisys::fp16_t *>(vals), numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(vals_type);
    }
}

} // namespace llaisys::ops::cpu
