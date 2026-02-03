#include "llaisys/models/qwen2.h"
#include "../models/qwen2/qwen2.hpp"

__C {
    struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device_type, int *device_ids, int ndevice) {
        int device_id = (ndevice > 0 && device_ids != nullptr) ? device_ids[0] : 0;
        auto model = new llaisys::models::Qwen2(*meta, device_type, device_id);
        return reinterpret_cast<struct LlaisysQwen2Model *>(model);
    }

    void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model * model) {
        delete reinterpret_cast<llaisys::models::Qwen2 *>(model);
    }

    struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model * model) {
        return reinterpret_cast<llaisys::models::Qwen2 *>(model)->weights();
    }

    // 更新：参数包含 pos
    int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model * model, int64_t * token_ids, size_t ntoken, size_t pos) {
        return reinterpret_cast<llaisys::models::Qwen2 *>(model)->infer(token_ids, ntoken, pos);
    }
}