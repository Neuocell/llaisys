#include "qwen2.hpp"
// 新增：引入 LlaisysTensor 的完整定义
#include "../../llaisys/llaisys_tensor.hpp" 

#include "../../ops/add/op.hpp"
#include "../../ops/argmax/op.hpp"
#include "../../ops/embedding/op.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/rms_norm/op.hpp"
#include "../../ops/rope/op.hpp"
#include "../../ops/self_attention/op.hpp"
#include "../../ops/swiglu/op.hpp"
#include "../../core/context/context.hpp"
#include <cmath>

namespace llaisys::models {

using namespace llaisys::ops;

Qwen2::Qwen2(const LlaisysQwen2Meta &meta, llaisysDeviceType_t device_type, int device_id)
    : _meta(meta), _device_type(device_type), _device_id(device_id) {
    
    core::context().setDevice(_device_type, _device_id);

    // Resize storage vectors
    _attn_norm_w_storage.resize(meta.nlayer);
    _attn_q_w_storage.resize(meta.nlayer);
    _attn_q_b_storage.resize(meta.nlayer);
    _attn_k_w_storage.resize(meta.nlayer);
    _attn_k_b_storage.resize(meta.nlayer);
    _attn_v_w_storage.resize(meta.nlayer);
    _attn_v_b_storage.resize(meta.nlayer);
    _attn_o_w_storage.resize(meta.nlayer);
    _mlp_norm_w_storage.resize(meta.nlayer);
    _mlp_gate_w_storage.resize(meta.nlayer);
    _mlp_up_w_storage.resize(meta.nlayer);
    _mlp_down_w_storage.resize(meta.nlayer);

    _weights.attn_norm_w = _attn_norm_w_storage.data();
    _weights.attn_q_w = _attn_q_w_storage.data();
    _weights.attn_q_b = _attn_q_b_storage.data();
    _weights.attn_k_w = _attn_k_w_storage.data();
    _weights.attn_k_b = _attn_k_b_storage.data();
    _weights.attn_v_w = _attn_v_w_storage.data();
    _weights.attn_v_b = _attn_v_b_storage.data();
    _weights.attn_o_w = _attn_o_w_storage.data();
    _weights.mlp_norm_w = _mlp_norm_w_storage.data();
    _weights.mlp_gate_w = _mlp_gate_w_storage.data();
    _weights.mlp_up_w = _mlp_up_w_storage.data();
    _weights.mlp_down_w = _mlp_down_w_storage.data();

    size_t head_dim = meta.di / meta.nh;
    
    // Create wrapper tensors
    _weights.in_embed = create_tensor_wrapper(new_tensor({meta.voc, meta.di}));
    _weights.out_embed = create_tensor_wrapper(new_tensor({meta.voc, meta.di}));
    _weights.out_norm_w = create_tensor_wrapper(new_tensor({meta.di}));

    for (size_t i = 0; i < meta.nlayer; ++i) {
        _weights.attn_norm_w[i] = create_tensor_wrapper(new_tensor({meta.di}));
        
        _weights.attn_q_w[i] = create_tensor_wrapper(new_tensor({meta.nh * head_dim, meta.di}));
        _weights.attn_q_b[i] = create_tensor_wrapper(new_tensor({meta.nh * head_dim}));
        _weights.attn_k_w[i] = create_tensor_wrapper(new_tensor({meta.nkvh * head_dim, meta.di}));
        _weights.attn_k_b[i] = create_tensor_wrapper(new_tensor({meta.nkvh * head_dim}));
        _weights.attn_v_w[i] = create_tensor_wrapper(new_tensor({meta.nkvh * head_dim, meta.di}));
        _weights.attn_v_b[i] = create_tensor_wrapper(new_tensor({meta.nkvh * head_dim}));
        _weights.attn_o_w[i] = create_tensor_wrapper(new_tensor({meta.di, meta.nh * head_dim}));
        
        _weights.mlp_norm_w[i] = create_tensor_wrapper(new_tensor({meta.di}));
        _weights.mlp_gate_w[i] = create_tensor_wrapper(new_tensor({meta.hs, meta.di}));
        _weights.mlp_up_w[i] = create_tensor_wrapper(new_tensor({meta.hs, meta.di}));
        _weights.mlp_down_w[i] = create_tensor_wrapper(new_tensor({meta.di, meta.hs}));
        
        // Initialize KV Cache
        auto k_cache = new_tensor({meta.maxseq, meta.nkvh, head_dim});
        auto v_cache = new_tensor({meta.maxseq, meta.nkvh, head_dim});
        _kv_cache.push_back({k_cache, v_cache});
    }
}

Qwen2::~Qwen2() {
    // 这里的 delete 现在可以正常工作了，因为编译器看到了结构体定义
    delete _weights.in_embed;
    delete _weights.out_embed;
    delete _weights.out_norm_w;
    for(auto t : _attn_norm_w_storage) delete t;
    for(auto t : _attn_q_w_storage) delete t;
    for(auto t : _attn_q_b_storage) delete t;
    for(auto t : _attn_k_w_storage) delete t;
    for(auto t : _attn_k_b_storage) delete t;
    for(auto t : _attn_v_w_storage) delete t;
    for(auto t : _attn_v_b_storage) delete t;
    for(auto t : _attn_o_w_storage) delete t;
    for(auto t : _mlp_norm_w_storage) delete t;
    for(auto t : _mlp_gate_w_storage) delete t;
    for(auto t : _mlp_up_w_storage) delete t;
    for(auto t : _mlp_down_w_storage) delete t;
}

tensor_t Qwen2::new_tensor(const std::vector<size_t>& shape) {
    return Tensor::create(shape, _meta.dtype, _device_type, _device_id);
}

llaisysTensor_t Qwen2::create_tensor_wrapper(tensor_t t) {
    return new LlaisysTensor{t};
}

int64_t Qwen2::infer(int64_t *token_ids, size_t ntoken, size_t pos) {
    core::context().setDevice(_device_type, _device_id);

    size_t seq_len = ntoken;
    size_t head_dim = _meta.di / _meta.nh;
    
    // Inputs
    auto input_ids_t = Tensor::create({seq_len}, LLAISYS_DTYPE_I64, _device_type, _device_id);
    auto pos_ids_t = Tensor::create({seq_len}, LLAISYS_DTYPE_I64, _device_type, _device_id);
    
    input_ids_t->load(token_ids);
    std::vector<int64_t> pos_vec(seq_len);
    for(size_t i=0; i<seq_len; ++i) pos_vec[i] = pos + i;
    pos_ids_t->load(pos_vec.data());

    // 1. Embedding
    auto hidden_states = new_tensor({seq_len, _meta.di});
    embedding(hidden_states, input_ids_t, _weights.in_embed->tensor);

    // 2. Layers
    for (size_t i = 0; i < _meta.nlayer; ++i) {
        auto residual = hidden_states;
        auto norm_out = new_tensor({seq_len, _meta.di});
        
        // Attention Block
        rms_norm(norm_out, hidden_states, _weights.attn_norm_w[i]->tensor, _meta.epsilon);
        
        auto q = new_tensor({seq_len, _meta.nh * head_dim});
        auto k = new_tensor({seq_len, _meta.nkvh * head_dim});
        auto v = new_tensor({seq_len, _meta.nkvh * head_dim});
        
        linear(q, norm_out, _weights.attn_q_w[i]->tensor, _weights.attn_q_b[i]->tensor);
        linear(k, norm_out, _weights.attn_k_w[i]->tensor, _weights.attn_k_b[i]->tensor);
        linear(v, norm_out, _weights.attn_v_w[i]->tensor, _weights.attn_v_b[i]->tensor);

        q = q->view({seq_len, _meta.nh, head_dim});
        k = k->view({seq_len, _meta.nkvh, head_dim});
        v = v->view({seq_len, _meta.nkvh, head_dim});

        rope(q, q, pos_ids_t, _meta.theta);
        rope(k, k, pos_ids_t, _meta.theta);

        // Update KV Cache
        auto& k_cache = _kv_cache[i].first;
        auto& v_cache = _kv_cache[i].second;
        
        auto k_slot = k_cache->slice(0, pos, pos + seq_len);
        auto v_slot = v_cache->slice(0, pos, pos + seq_len);
        
        core::context().runtime().api()->memcpy_sync(
            k_slot->data(), k->data(), k->numel() * k->elementSize(), LLAISYS_MEMCPY_D2D);
        core::context().runtime().api()->memcpy_sync(
            v_slot->data(), v->data(), v->numel() * v->elementSize(), LLAISYS_MEMCPY_D2D);
        
        // Full KV for attention
        auto k_full = k_cache->slice(0, 0, pos + seq_len);
        auto v_full = v_cache->slice(0, 0, pos + seq_len);

        // Attention
        auto attn_out = new_tensor({seq_len, _meta.nh, head_dim});
        float scale = 1.0f / sqrtf((float)head_dim);
        self_attention(attn_out, q, k_full, v_full, scale);

        attn_out = attn_out->view({seq_len, _meta.di});
        auto linear_out = new_tensor({seq_len, _meta.di});
        linear(linear_out, attn_out, _weights.attn_o_w[i]->tensor, nullptr);

        add(hidden_states, residual, linear_out);

        // MLP Block
        residual = hidden_states;
        rms_norm(norm_out, hidden_states, _weights.mlp_norm_w[i]->tensor, _meta.epsilon);
        
        auto gate = new_tensor({seq_len, _meta.hs});
        auto up = new_tensor({seq_len, _meta.hs});
        linear(gate, norm_out, _weights.mlp_gate_w[i]->tensor, nullptr);
        linear(up, norm_out, _weights.mlp_up_w[i]->tensor, nullptr);
        
        swiglu(gate, gate, up);
        
        auto down_out = new_tensor({seq_len, _meta.di});
        linear(down_out, gate, _weights.mlp_down_w[i]->tensor, nullptr);

        add(hidden_states, residual, down_out);
    }

    // 3. Final Norm
    rms_norm(hidden_states, hidden_states, _weights.out_norm_w->tensor, _meta.epsilon);

    // 4. Head
    auto last_hidden = hidden_states->slice(0, seq_len - 1, seq_len);
    auto logits = new_tensor({1, _meta.voc});
    linear(logits, last_hidden, _weights.out_embed->tensor, nullptr); 

    // 5. Argmax
    auto max_idx = Tensor::create({1}, LLAISYS_DTYPE_I64, _device_type, _device_id);
    auto max_val = new_tensor({1});
    argmax(max_idx, max_val, logits);
    
    int64_t result_token;
    core::context().runtime().api()->memcpy_sync(
        &result_token, max_idx->data(), sizeof(int64_t), LLAISYS_MEMCPY_D2H);

    return result_token;
}

} // namespace llaisys::models