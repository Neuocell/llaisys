import ctypes
from .llaisys_types import llaisysDataType_t, llaisysDeviceType_t
from .tensor import llaisysTensor_t

# 1. 定义元数据结构体
class LlaisysQwen2Meta(ctypes.Structure):
    _fields_ = [
        ("dtype", llaisysDataType_t),
        ("nlayer", ctypes.c_size_t),
        ("hs", ctypes.c_size_t),
        ("nh", ctypes.c_size_t),
        ("nkvh", ctypes.c_size_t),
        ("dh", ctypes.c_size_t),
        ("di", ctypes.c_size_t),
        ("maxseq", ctypes.c_size_t),
        ("voc", ctypes.c_size_t),
        ("epsilon", ctypes.c_float),
        ("theta", ctypes.c_float),
        ("end_token", ctypes.c_int64),
    ]

# 2. 定义权重结构体
class LlaisysQwen2Weights(ctypes.Structure):
    _fields_ = [
        ("in_embed", llaisysTensor_t),
        ("out_embed", llaisysTensor_t),
        ("out_norm_w", llaisysTensor_t),
        ("attn_norm_w", ctypes.POINTER(llaisysTensor_t)),
        ("attn_q_w", ctypes.POINTER(llaisysTensor_t)),
        ("attn_q_b", ctypes.POINTER(llaisysTensor_t)),
        ("attn_k_w", ctypes.POINTER(llaisysTensor_t)),
        ("attn_k_b", ctypes.POINTER(llaisysTensor_t)),
        ("attn_v_w", ctypes.POINTER(llaisysTensor_t)),
        ("attn_v_b", ctypes.POINTER(llaisysTensor_t)),
        ("attn_o_w", ctypes.POINTER(llaisysTensor_t)),
        ("mlp_norm_w", ctypes.POINTER(llaisysTensor_t)),
        ("mlp_gate_w", ctypes.POINTER(llaisysTensor_t)),
        ("mlp_up_w", ctypes.POINTER(llaisysTensor_t)),
        ("mlp_down_w", ctypes.POINTER(llaisysTensor_t)),
    ]

llaisysQwen2Model_t = ctypes.c_void_p

# 3. 注册函数签名的加载函数
def load_models(lib):
    if hasattr(lib, 'llaisysQwen2ModelCreate'):
        lib.llaisysQwen2ModelCreate.argtypes = [
            ctypes.POINTER(LlaisysQwen2Meta),
            llaisysDeviceType_t,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int
        ]
        lib.llaisysQwen2ModelCreate.restype = llaisysQwen2Model_t

    if hasattr(lib, 'llaisysQwen2ModelDestroy'):
        lib.llaisysQwen2ModelDestroy.argtypes = [llaisysQwen2Model_t]
        lib.llaisysQwen2ModelDestroy.restype = None

    if hasattr(lib, 'llaisysQwen2ModelWeights'):
        lib.llaisysQwen2ModelWeights.argtypes = [llaisysQwen2Model_t]
        # 关键修复：指定返回类型为指针，而不是默认的 int
        lib.llaisysQwen2ModelWeights.restype = ctypes.POINTER(LlaisysQwen2Weights)

    if hasattr(lib, 'llaisysQwen2ModelInfer'):
        lib.llaisysQwen2ModelInfer.argtypes = [
            llaisysQwen2Model_t, 
            ctypes.POINTER(ctypes.c_int64), 
            ctypes.c_size_t, 
            ctypes.c_size_t # pos 参数
        ]
        lib.llaisysQwen2ModelInfer.restype = ctypes.c_int64