#include "embedding_cpu.hpp"

#include "../../../utils.hpp"
#include <cstdint>

// Embedding 核心计算模板
template <typename T, typename IndexT>
void embedding_(T *out, const IndexT *index, const T *weight, 
                size_t index_size, size_t embedding_dim) {
    for (size_t i = 0; i < index_size; i++) {
        IndexT idx = index[i];
        const T *src = weight + idx * embedding_dim;
        T *dst = out + i * embedding_dim;
        
        for (size_t j = 0; j < embedding_dim; j++) {
            dst[j] = src[j];
        }
    }
}

namespace llaisys::ops::cpu {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight, 
               llaisysDataType_t weight_type, llaisysDataType_t index_type, 
               size_t index_size, size_t embedding_dim) {
    // 处理索引类型
    switch (index_type) {
    case LLAISYS_DTYPE_I32:
        // 处理权重类型
        switch (weight_type) {
        case LLAISYS_DTYPE_F32:
            return embedding_(reinterpret_cast<float *>(out), 
                             reinterpret_cast<const int32_t *>(index),
                             reinterpret_cast<const float *>(weight), 
                             index_size, embedding_dim);
        case LLAISYS_DTYPE_BF16:
            return embedding_(reinterpret_cast<llaisys::bf16_t *>(out), 
                             reinterpret_cast<const int32_t *>(index),
                             reinterpret_cast<const llaisys::bf16_t *>(weight), 
                             index_size, embedding_dim);
        case LLAISYS_DTYPE_F16:
            return embedding_(reinterpret_cast<llaisys::fp16_t *>(out), 
                             reinterpret_cast<const int32_t *>(index),
                             reinterpret_cast<const llaisys::fp16_t *>(weight), 
                             index_size, embedding_dim);
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(weight_type);
        }
        break;
    case LLAISYS_DTYPE_I64:
        // 处理权重类型
        switch (weight_type) {
        case LLAISYS_DTYPE_F32:
            return embedding_(reinterpret_cast<float *>(out), 
                             reinterpret_cast<const int64_t *>(index),
                             reinterpret_cast<const float *>(weight), 
                             index_size, embedding_dim);
        case LLAISYS_DTYPE_BF16:
            return embedding_(reinterpret_cast<llaisys::bf16_t *>(out), 
                             reinterpret_cast<const int64_t *>(index),
                             reinterpret_cast<const llaisys::bf16_t *>(weight), 
                             index_size, embedding_dim);
        case LLAISYS_DTYPE_F16:
            return embedding_(reinterpret_cast<llaisys::fp16_t *>(out), 
                             reinterpret_cast<const int64_t *>(index),
                             reinterpret_cast<const llaisys::fp16_t *>(weight), 
                             index_size, embedding_dim);
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(weight_type);
        }
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(index_type);
    }
}
} // namespace llaisys::ops::cpu
