/**
 * Copyright 2022 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "plugin/device/gpu/kernel/nn/resize_linear_1d_gpu_kernel.h"
#include "mindspore/core/abstract/utils.h"

namespace {
constexpr const size_t kResizeLinear1DInputsNum = 2;
constexpr const size_t kResizeLinear1DOutputsNum = 1;
constexpr const size_t kResizeInputDims = 3;
}  // namespace

namespace mindspore {
namespace kernel {
void ResizeLinear1DGpuKernelMod::InitSizeLists() {
  input_size_list_.push_back(input_byte_size_);
  output_size_list_.push_back(output_byte_size_);
}

void ResizeLinear1DGpuKernelMod::GetSize() {
  input_byte_size_ = input_data_unit_size_;
  for (const auto &shape_item : input_shape_) {
    input_byte_size_ *= shape_item;
  }
  output_byte_size_ = output_data_unit_size_;
  for (const auto &shape_item : output_shape_) {
    output_byte_size_ *= shape_item;
  }
}

bool ResizeLinear1DGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                      const std::vector<KernelTensorPtr> &outputs) {
  MS_ERROR_IF_NULL_W_RET_VAL(base_operator, false);

  auto kernel_ptr = std::dynamic_pointer_cast<ops::ResizeLinear1D>(base_operator);
  MS_ERROR_IF_NULL_W_RET_VAL(kernel_ptr, false);

  kernel_name_ = kernel_ptr->name();
  if (inputs.size() != kResizeLinear1DInputsNum || outputs.size() != kResizeLinear1DOutputsNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', input and output size must be " << kResizeLinear1DInputsNum
                  << " and " << kResizeLinear1DOutputsNum << ", but got " << inputs.size() << " and " << outputs.size();
    return false;
  }

  std::string coordinate_transformation_mode = kernel_ptr->get_coordinate_transformation_mode();
  if (coordinate_transformation_mode == "align_corners") {
    mode_ = ResizeLinearCoordinateTransformationMode::ALIGN_CORNERS;
  } else if (coordinate_transformation_mode == "half_pixel") {
    mode_ = ResizeLinearCoordinateTransformationMode::HALF_PIXEL;
  } else if (coordinate_transformation_mode == "asymmetric") {
    mode_ = ResizeLinearCoordinateTransformationMode::ASYMMETRIC;
  } else {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', coordinate_transformation_mode: " << coordinate_transformation_mode
                  << " not support now.";
    return false;
  }

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto pair = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!pair.first) {
    MS_LOG(ERROR) << "'" << kernel_name_ << "' does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[pair.second].second;

  input_data_unit_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex0).first);
  output_data_unit_size_ = abstract::TypeIdSize(kernel_attr.GetOutputAttr(kIndex0).first);

  return true;
}

void ResizeLinear1DGpuKernelMod::ResetResource() {
  input_size_list_.clear();
  output_size_list_.clear();
}

int ResizeLinear1DGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                       const std::vector<KernelTensorPtr> &outputs,
                                       const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  int ret;
  if ((ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost)) != KRET_OK) {
    return ret;
  }
  ResetResource();

  input_shape_ = std::vector<int64_t>(inputs.at(kIndex0)->GetDeviceShapeAdaptively().begin(),
                                      inputs.at(kIndex0)->GetDeviceShapeAdaptively().end());
  batch_ = LongToSize(input_shape_[kIndex0]);
  channel_ = LongToSize(input_shape_[kIndex1]);
  in_width_ = input_shape_[kIndex2];

  output_shape_ = std::vector<int64_t>(outputs.at(kIndex0)->GetDeviceShapeAdaptively().begin(),
                                       outputs.at(kIndex0)->GetDeviceShapeAdaptively().end());

  out_width_ = output_shape_[kIndex2];

  if (input_shape_.size() != kResizeInputDims) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the dimension of 'input_x' should be greater than or equal to 1, but got "
                  << input_shape_.size() << ".";
    return KRET_RESIZE_FAILED;
  }
  GetSize();
  InitSizeLists();
  return KRET_OK;
}

template <typename T>
bool ResizeLinear1DGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                              const std::vector<AddressPtr> &workspace,
                                              const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kResizeLinear1DInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kResizeLinear1DOutputsNum, kernel_name_);
  T *input = GetDeviceAddress<T>(inputs, kIndex0);
  MS_ERROR_IF_NULL_W_RET_VAL(input, false);

  float *output = GetDeviceAddress<float>(outputs, kIndex0);
  MS_ERROR_IF_NULL_W_RET_VAL(output, false);

  int64_t output_size = batch_ * channel_ * out_width_;

  ResizeLinear1D(mode_, output_size, in_width_, out_width_, input, output, device_id_,
                 reinterpret_cast<cudaStream_t>(stream_ptr));

  return true;
}

#define RESIZE_LINEAR_1D_GPU_REG(MS_T, T)                                                           \
  KernelAttr().AddInputAttr(MS_T).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat32), \
    &ResizeLinear1DGpuKernelMod::LaunchKernel<T>

std::vector<std::pair<KernelAttr, ResizeLinear1DGpuKernelMod::ResizeLinear1DFunc>>
  ResizeLinear1DGpuKernelMod::func_list_ = {
    {RESIZE_LINEAR_1D_GPU_REG(kNumberTypeInt8, int8_t)},    {RESIZE_LINEAR_1D_GPU_REG(kNumberTypeUInt8, uint8_t)},
    {RESIZE_LINEAR_1D_GPU_REG(kNumberTypeInt16, int16_t)},  {RESIZE_LINEAR_1D_GPU_REG(kNumberTypeUInt16, uint16_t)},
    {RESIZE_LINEAR_1D_GPU_REG(kNumberTypeInt32, int32_t)},  {RESIZE_LINEAR_1D_GPU_REG(kNumberTypeInt64, int64_t)},
    {RESIZE_LINEAR_1D_GPU_REG(kNumberTypeFloat16, half)},   {RESIZE_LINEAR_1D_GPU_REG(kNumberTypeFloat32, float)},
    {RESIZE_LINEAR_1D_GPU_REG(kNumberTypeFloat64, double)},
};

std::vector<KernelAttr> ResizeLinear1DGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, ResizeLinear1DFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, ResizeLinear1D, ResizeLinear1DGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
