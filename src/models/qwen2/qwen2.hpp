#pragma once
#include "llaisys/models/qwen2.h"
#include "../../tensor/tensor.hpp"
#include <vector>

namespace llaisys::models {

class Qwen2 {
public:
    Qwen2(const LlaisysQwen2Meta &meta, llaisysDeviceType_t device_type, int device_id);
    ~Qwen2();

    LlaisysQwen2Weights *weights() { return &_weights; }
    
    // 更新：增加 pos 参数
    int64_t infer(int64_t *token_ids, size_t ntoken, size_t pos);

private:
    LlaisysQwen2Meta _meta;
    llaisysDeviceType_t _device_type;
    int _device_id;

    LlaisysQwen2Weights _weights;
    
    // KV Cache: [layer][k/v]
    std::vector<std::pair<tensor_t, tensor_t>> _kv_cache;
    
    // 移除 _cur_pos，因为位置现在由调用者管理

    // 权重存储容器
    std::vector<llaisysTensor_t> _attn_norm_w_storage;
    std::vector<llaisysTensor_t> _attn_q_w_storage;
    std::vector<llaisysTensor_t> _attn_q_b_storage;
    std::vector<llaisysTensor_t> _attn_k_w_storage;
    std::vector<llaisysTensor_t> _attn_k_b_storage;
    std::vector<llaisysTensor_t> _attn_v_w_storage;
    std::vector<llaisysTensor_t> _attn_v_b_storage;
    std::vector<llaisysTensor_t> _attn_o_w_storage;
    std::vector<llaisysTensor_t> _mlp_norm_w_storage;
    std::vector<llaisysTensor_t> _mlp_gate_w_storage;
    std::vector<llaisysTensor_t> _mlp_up_w_storage;
    std::vector<llaisysTensor_t> _mlp_down_w_storage;

    llaisysTensor_t create_tensor_wrapper(tensor_t t);
    tensor_t new_tensor(const std::vector<size_t>& shape);
};

} // namespace llaisys::models