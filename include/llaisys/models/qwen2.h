// llaisys/include/llaisys/models/qwen2.h

#ifndef LLAISYS_MODELS_QWEN2_H
#define LLAISYS_MODELS_QWEN2_H

#include "../tensor.h"

__C {
    struct LlaisysQwen2Meta {
        llaisysDataType_t dtype;
        size_t nlayer, hs, nh, nkvh, dh, di, maxseq, voc;
        float epsilon, theta;
        int64_t end_token;
    };

    struct LlaisysQwen2Weights {
        llaisysTensor_t in_embed;    // model.embed_tokens.weight
        llaisysTensor_t out_embed;   // lm_head.weight
        llaisysTensor_t out_norm_w;  // model.norm.weight
        
        // 以下为每层的权重数组
        llaisysTensor_t *attn_norm_w; // input_layernorm.weight
        llaisysTensor_t *attn_q_w;    // self_attn.q_proj.weight
        llaisysTensor_t *attn_q_b;    // self_attn.q_proj.bias
        llaisysTensor_t *attn_k_w;    // self_attn.k_proj.weight
        llaisysTensor_t *attn_k_b;    // self_attn.k_proj.bias
        llaisysTensor_t *attn_v_w;    // self_attn.v_proj.weight
        llaisysTensor_t *attn_v_b;    // self_attn.v_proj.bias
        llaisysTensor_t *attn_o_w;    // self_attn.o_proj.weight
        
        llaisysTensor_t *mlp_norm_w;  // post_attention_layernorm.weight
        llaisysTensor_t *mlp_gate_w;  // mlp.gate_proj.weight
        llaisysTensor_t *mlp_up_w;    // mlp.up_proj.weight
        llaisysTensor_t *mlp_down_w;  // mlp.down_proj.weight
    };

    struct LlaisysQwen2Model;

    __export struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice);
    __export void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model * model);
    __export struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model * model);
    // 增加一个设置当前位置的接口，用于推理循环
    __export int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model * model, int64_t * token_ids, size_t ntoken, size_t pos);
}
#endif