from typing import Sequence
from ..libllaisys import LIB_LLAISYS
from ..libllaisys import DeviceType, DataType
from ..libllaisys.models import LlaisysQwen2Meta, LlaisysQwen2Weights
from ..tensor import Tensor
import ctypes
from pathlib import Path
import safetensors
import json
import torch
import time

class Qwen2:
    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        self.model_path = Path(model_path)
        self.device = device
        
        # 1. Load Config
        with open(self.model_path / "config.json", "r") as f:
            config = json.load(f)
        
        # 2. Prepare Meta
        self.meta = LlaisysQwen2Meta()
        self.meta.dtype = DataType.BF16
        self.meta.nlayer = config["num_hidden_layers"]
        self.meta.hs = config["intermediate_size"]
        self.meta.nh = config["num_attention_heads"]
        self.meta.nkvh = config["num_key_value_heads"]
        self.meta.di = config["hidden_size"]
        self.meta.dh = self.meta.di // self.meta.nh
        self.meta.maxseq = 4096 
        self.meta.voc = config["vocab_size"]
        self.meta.epsilon = config["rms_norm_eps"]
        self.meta.theta = config.get("rope_theta", 1000000.0)
        self.meta.end_token = config.get("eos_token_id", 151643)

        # 3. Create Model
        device_ids = (ctypes.c_int * 1)(0)
        self._model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            ctypes.byref(self.meta), 
            device, 
            device_ids, 
            1
        )
        
        # 4. Get Weights Structure
        self._weights = LIB_LLAISYS.llaisysQwen2ModelWeights(self._model).contents

        # 5. Load Weights
        print("Loading weights...", flush=True)
        self._load_weights()
        print("Weights loaded.", flush=True)

    def _load_weights(self):
        for file in sorted(self.model_path.glob("*.safetensors")):
            with safetensors.safe_open(file, framework="pt", device="cpu") as f:
                for name in f.keys():
                    data = f.get_tensor(name)
                    self._load_tensor(name, data)

    def _load_tensor(self, name, data):
        tensor_ptr = None
        
        if "lm_head.weight" in name:
            tensor_ptr = self._weights.out_embed
        elif "model.embed_tokens.weight" in name:
            tensor_ptr = self._weights.in_embed
        elif "model.norm.weight" in name:
            tensor_ptr = self._weights.out_norm_w
        elif "layers" in name:
            parts = name.split(".")
            idx = int(parts[2])
            
            if "input_layernorm.weight" in name:
                tensor_ptr = self._weights.attn_norm_w[idx]
            elif "post_attention_layernorm.weight" in name:
                tensor_ptr = self._weights.mlp_norm_w[idx]
            elif "self_attn" in name:
                if "q_proj.weight" in name: tensor_ptr = self._weights.attn_q_w[idx]
                elif "q_proj.bias" in name: tensor_ptr = self._weights.attn_q_b[idx]
                elif "k_proj.weight" in name: tensor_ptr = self._weights.attn_k_w[idx]
                elif "k_proj.bias" in name: tensor_ptr = self._weights.attn_k_b[idx]
                elif "v_proj.weight" in name: tensor_ptr = self._weights.attn_v_w[idx]
                elif "v_proj.bias" in name: tensor_ptr = self._weights.attn_v_b[idx]
                elif "o_proj.weight" in name: tensor_ptr = self._weights.attn_o_w[idx]
            elif "mlp" in name:
                if "gate_proj.weight" in name: tensor_ptr = self._weights.mlp_gate_w[idx]
                elif "up_proj.weight" in name: tensor_ptr = self._weights.mlp_up_w[idx]
                elif "down_proj.weight" in name: tensor_ptr = self._weights.mlp_down_w[idx]

        if tensor_ptr:
            t = Tensor(tensor=tensor_ptr)
            if not data.is_contiguous():
                data = data.contiguous()
            ptr = ctypes.c_void_p(data.data_ptr())
            t.load(ptr)
            t._tensor = None 

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = 128,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        # 修正：结果列表必须包含输入的 prompt tokens，以匹配 HF 的行为
        result = list(inputs)
        current_pos = 0 
        
        # Prefill
        tokens_buf = (ctypes.c_int64 * len(inputs))(*inputs)
        
        print(f"Start Prefill ({len(inputs)} tokens)...", end=" ", flush=True)
        t0 = time.time()
        # Prefill 阶段处理整个 prompt，返回第一个生成的 token
        next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(self._model, tokens_buf, len(inputs), current_pos)
        t1 = time.time()
        print(f"Done. Time: {(t1-t0)*1000:.2f} ms", flush=True)
        
        result.append(next_token)
        current_pos += len(inputs)
        
        # Decode
        print("Start Decoding...", flush=True)
        for i in range(max_new_tokens - 1):
            if next_token == self.meta.end_token:
                print("\n[EOS Reached]", flush=True)
                break
            
            tokens_buf = (ctypes.c_int64 * 1)(next_token)
            
            t0 = time.time()
            next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(self._model, tokens_buf, 1, current_pos)
            t1 = time.time()
            
            print(f"\r[Decode] Step {i+1}/{max_new_tokens-1}: {(t1-t0)*1000:.2f} ms", end="", flush=True)
            
            result.append(next_token)
            current_pos += 1
        
        print("\nGeneration finished.", flush=True)
        return result