#include "tensor.hpp"

#include "../utils.hpp"

#include <cstring>
#include <numeric>
#include <sstream>

namespace llaisys {

Tensor::Tensor(TensorMeta meta, core::storage_t storage, size_t offset)
    : _meta(std::move(meta)), _storage(std::move(storage)), _offset(offset) {}
//llaisys/include/llaisys.h 
//llaisysDataType_t 和 llaisysDeviceType_t 是自定义枚举类型，分别用于表示数据类型和设备类型。
tensor_t Tensor::create(const std::vector<size_t> &shape,
                        llaisysDataType_t dtype,
                        llaisysDeviceType_t device_type,
                        int device) {
    size_t ndim_ = shape.size();
    //ptrdiff_t 是标准 C++ 类型，用于表示指针差值，定义在 <cstddef> 头文件中。
    // 它通常用于计算数组索引或指针偏移，确保在不同平台上的一致性。
    std::vector<ptrdiff_t> strides(ndim_);
    size_t stride = 1;
    for (size_t i = 1; i <= ndim_; i++) {
        strides[ndim_ - i] = stride;
        stride *= shape[ndim_ - i];
    }
    TensorMeta meta{dtype, shape, strides};
    size_t total_elems = stride;
    size_t dtype_size = utils::dsize(dtype);
//如果请求的 device_type 是 LLAISYS_DEVICE_CPU（CPU），
// 且当前运行时（core::context().runtime()）的设备类型不是 CPU（即在 GPU 或其他设备上），则执行特殊处理。
    if (device_type == LLAISYS_DEVICE_CPU && core::context().runtime().deviceType() != LLAISYS_DEVICE_CPU) {
        //调用 allocateHostStorage() 分配主机（CPU）存储，然后创建张量。
        auto storage = core::context().runtime().allocateHostStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));//只给两个参数，这对吗？？
    } else {
        //否则（请求非 CPU 设备，或当前已在 CPU 上），执行标准流程。
        core::context().setDevice(device_type, device);
        auto storage = core::context().runtime().allocateDeviceStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    }
}
//为什么这么做：避免在 GPU 上请求 CPU 张量时切换设备上下文（切换开销大），
// 直接在 CPU 分配内存。非 CPU 请求时，确保设备上下文正确设置。
//好处：减少设备切换延迟，提高创建效率；确保内存分配在正确设备上，防止访问错误内存导致崩溃。

std::byte *Tensor::data() {
    return _storage->memory() + _offset;
}

const std::byte *Tensor::data() const {
    return _storage->memory() + _offset;
}

size_t Tensor::ndim() const {
    return _meta.shape.size();
}

const std::vector<size_t> &Tensor::shape() const {
    return _meta.shape;
}

const std::vector<ptrdiff_t> &Tensor::strides() const {
    return _meta.strides;
}

llaisysDataType_t Tensor::dtype() const {
    return _meta.dtype;
}

llaisysDeviceType_t Tensor::deviceType() const {
    return _storage->deviceType();
}

int Tensor::deviceId() const {
    return _storage->deviceId();
}

size_t Tensor::numel() const {
    return std::accumulate(_meta.shape.begin(), _meta.shape.end(), size_t(1), std::multiplies<size_t>());
}

size_t Tensor::elementSize() const {
    return utils::dsize(_meta.dtype);
}

std::string Tensor::info() const {
    std::stringstream ss;

    ss << "Tensor: "
       << "shape[ ";
    for (auto s : this->shape()) {
        ss << s << " ";
    }
    ss << "] strides[ ";
    for (auto s : this->strides()) {
        ss << s << " ";
    }
    ss << "] dtype=" << this->dtype();

    return ss.str();
}

template <typename T>
void print_data(const T *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, size_t dim) {
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                std::cout << utils::cast<float>(data[i * strides[dim]]) << " ";
            } else {
                std::cout << data[i * strides[dim]] << " ";
            }
        }
        std::cout << std::endl;
    } else if (dim < shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            print_data(data + i * strides[dim], shape, strides, dim + 1);
        }
    }
}//递归写法

void debug_print(const std::byte *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_BYTE:
        return print_data(reinterpret_cast<const char *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BOOL:
        return print_data(reinterpret_cast<const bool *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I8:
        return print_data(reinterpret_cast<const int8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I16:
        return print_data(reinterpret_cast<const int16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I32:
        return print_data(reinterpret_cast<const int32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I64:
        return print_data(reinterpret_cast<const int64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U8:
        return print_data(reinterpret_cast<const uint8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U16:
        return print_data(reinterpret_cast<const uint16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U32:
        return print_data(reinterpret_cast<const uint32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U64:
        return print_data(reinterpret_cast<const uint64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F16:
        return print_data(reinterpret_cast<const fp16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F32:
        return print_data(reinterpret_cast<const float *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F64:
        return print_data(reinterpret_cast<const double *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BF16:
        return print_data(reinterpret_cast<const bf16_t *>(data), shape, strides, 0);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void Tensor::debug() const {
    core::context().setDevice(this->deviceType(), this->deviceId());
    core::context().runtime().api()->device_synchronize();
    std::cout << this->info() << std::endl;
    //设置设备到张量所在设备，同步设备操作，然后打印 info。
    // 如果在 CPU，直接调用 debug_print；
    // 否则，创建临时 CPU 张量，拷贝数据到主机（D2H），再打印。
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        debug_print(this->data(), this->shape(), this->strides(), this->dtype());
    } else {
        auto tmp_tensor = create({this->_storage->size()}, this->dtype());
        core::context().runtime().api()->memcpy_sync(
            tmp_tensor->data(),
            this->data(),
            this->numel() * this->elementSize(),
            LLAISYS_MEMCPY_D2H);
        debug_print(tmp_tensor->data(), this->shape(), this->strides(), this->dtype());
    }
}


bool Tensor::isContiguous() const {
    ptrdiff_t expected_stride = 1;  // 改为 ptrdiff_t 类型
    for (size_t i = _meta.shape.size(); i-- > 0;) {
        if (_meta.strides[i] != expected_stride) {
            return false;
        }
        expected_stride *= static_cast<ptrdiff_t>(_meta.shape[i]);  // 确保乘法结果也是 ptrdiff_t
    }
    return true;
}
//维度顺序定义了坐标轴的排列。
// permute 根据给定的 order（维度索引的排列）重新排列这些轴，
// 而不改变底层数据的值。
tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    CHECK_ARGUMENT(order.size() == _meta.shape.size(), "Permute order size must match tensor dimension");
    
    std::vector<size_t> new_shape(order.size());
    std::vector<ptrdiff_t> new_strides(order.size());
    
    for (size_t i = 0; i < order.size(); i++) {
        CHECK_ARGUMENT(order[i] < order.size(), "Invalid dimension index in permute order");
        new_shape[i] = _meta.shape[order[i]];
        new_strides[i] = _meta.strides[order[i]];
    }
    
    TensorMeta new_meta{_meta.dtype, new_shape, new_strides};
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, _offset));
}
tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    CHECK_ARGUMENT(dim < _meta.shape.size(), "Slice dimension out of range");
    CHECK_ARGUMENT(start <= end && end <= _meta.shape[dim], "Invalid slice range");
    
    std::vector<size_t> new_shape = _meta.shape;
    new_shape[dim] = end - start;
    
    TensorMeta new_meta{_meta.dtype, new_shape, _meta.strides};
    
    // 修复：将 _meta.strides[dim] 转换为 size_t
    size_t new_offset = _offset + start * static_cast<size_t>(_meta.strides[dim]) * elementSize();
    //_offset 表示当前张量的起始字节偏移。
    // 沿 dim 维度，strides[dim] 定义步长（元素数），表示移动一个单位在该维度跳过的元素。
    
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, new_offset));
}

tensor_t Tensor::view(const std::vector<size_t> &shape) const {
    size_t total_elements = std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>());
    CHECK_ARGUMENT(total_elements == numel(), "View shape must have same number of elements");
    
    std::vector<ptrdiff_t> new_strides(shape.size());
    ptrdiff_t stride = 1;
    for (size_t i = shape.size(); i-- > 0;) {
        new_strides[i] = stride;
        // 修复：将 shape[i] 转换为 ptrdiff_t
        stride *= static_cast<ptrdiff_t>(shape[i]);
    }
    
    TensorMeta new_meta{_meta.dtype, shape, new_strides};
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, _offset));
}

void Tensor::load(const void *src_) {
    core::context().setDevice(this->deviceType(), this->deviceId());
    
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        // CPU到CPU的直接拷贝
        std::memcpy(this->data(), src_, numel() * elementSize());
    } else {
        // 设备内存拷贝
        core::context().runtime().api()->memcpy_sync(
            this->data(),
            src_,
            numel() * elementSize(),
            LLAISYS_MEMCPY_H2D);
    }
}
tensor_t Tensor::contiguous() const {
    if (isContiguous()) {
        return std::shared_ptr<Tensor>(new Tensor(_meta, _storage, _offset));//为什么返回的是一个新向量的shared_ptr
    }
    
    // 创建新的连续张量（空的）
    auto new_tensor = create(_meta.shape, _meta.dtype, deviceType(), deviceId());
    
    // 复制数据
    core::context().setDevice(this->deviceType(), this->deviceId());
    core::context().runtime().api()->memcpy_sync(
        new_tensor->data(),
        this->data(),
        numel() * elementSize(),
        LLAISYS_MEMCPY_D2D);
    
    return new_tensor;
}
tensor_t Tensor::reshape(const std::vector<size_t> &shape) const {
    size_t total_elements = std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>());
    CHECK_ARGUMENT(total_elements == numel(), "Reshape must preserve total number of elements");
    
    if (isContiguous()) {
        // 连续张量可以直接创建视图
        return view(shape);
    } else {
        // 非连续张量需要先连续化
        auto contiguous_tensor = contiguous();
        return contiguous_tensor->view(shape);
    }
}
tensor_t Tensor::to(llaisysDeviceType_t device_type, int device) const {
    if (device_type == this->deviceType() && 
        (device == -1 || device == this->deviceId())) {
        // 相同设备，直接返回共享张量
        return std::shared_ptr<Tensor>(new Tensor(_meta, _storage, _offset));
    }
    
    // 创建目标设备上的新张量
    auto target_tensor = create(_meta.shape, _meta.dtype, device_type, device);
    
    // 执行数据传输
    core::context().setDevice(this->deviceType(), this->deviceId());
    core::context().runtime().api()->memcpy_sync(
        target_tensor->data(),
        this->data(),
        numel() * elementSize(),
        device_type == LLAISYS_DEVICE_CPU ? LLAISYS_MEMCPY_D2H : 
        (this->deviceType() == LLAISYS_DEVICE_CPU ? LLAISYS_MEMCPY_H2D : LLAISYS_MEMCPY_D2D));
    
    return target_tensor;
}


} // namespace llaisys
