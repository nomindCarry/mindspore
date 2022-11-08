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

#include "plugin/device/gpu/kernel/map_tensor/map_tensor_put_gpu_kernel.h"
#include <functional>
#include <utility>
#include <string>
#include "mindspore/core/abstract/utils.h"
#include "kernel/common_utils.h"

namespace mindspore {
namespace kernel {
std::vector<std::pair<KernelAttr, MapTensorPutGpuKernelMod::MapTensorPutLaunchFunc>>
  MapTensorPutGpuKernelMod::map_tensor_put_func_list_ = {{KernelAttr()
                                                            .AddInputAttr(kObjectTypeMapTensorType)
                                                            .AddInputAttr(kNumberTypeInt32)
                                                            .AddInputAttr(kNumberTypeFloat32)
                                                            .AddOutputAttr(kObjectTypeMapTensorType),
                                                          &MapTensorPutGpuKernelMod::LaunchKernel<int32_t, float>},
                                                         {KernelAttr()
                                                            .AddInputAttr(kObjectTypeMapTensorType)
                                                            .AddInputAttr(kNumberTypeInt64)
                                                            .AddInputAttr(kNumberTypeFloat32)
                                                            .AddOutputAttr(kObjectTypeMapTensorType),
                                                          &MapTensorPutGpuKernelMod::LaunchKernel<int64_t, float>}};

std::vector<KernelAttr> MapTensorPutGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    map_tensor_put_func_list_.begin(), map_tensor_put_func_list_.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, MapTensorPutGpuKernelMod::MapTensorPutLaunchFunc> &pair) { return pair.first; });
  return support_list;
}

bool MapTensorPutGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                    const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  MS_EXCEPTION_IF_NULL(base_operator->GetPrim());
  kernel_name_ = base_operator->GetPrim()->name();
  // Check the inputs and outputs num.
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kMapTensorPutInputNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kMapTensorPutOutputNum, kernel_name_);

  // Check the kernel attr.
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }

  // Get kernel launch function.
  kernel_launch_func_ = map_tensor_put_func_list_[index].second;
  InitSize(base_operator, inputs, outputs);
  return true;
}

int MapTensorPutGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                     const std::vector<KernelTensorPtr> &outputs,
                                     const std::map<uint32_t, tensor::TensorPtr> &) {
  ResetResource();
  InitSize(base_operator, inputs, outputs);
  return KRET_OK;
}

template <typename KeyType, typename ValueType>
bool MapTensorPutGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                            const std::vector<AddressPtr> &workspace,
                                            const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  // Check the inputs and outputs num.
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kMapTensorPutInputNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kMapTensorPutOutputNum, kernel_name_);

  // The real hash table should be accessed by user data.
  if (input_user_data_.empty()) {
    MS_LOG(EXCEPTION) << "The hash table user data is not set yet.";
  }

  auto user_data = input_user_data_[kIndex0];
  MS_EXCEPTION_IF_NULL(user_data);
  auto hash_table_ptr = user_data->get<GPUHashTable<KeyType, ValueType>>(kUserDataData);
  MS_EXCEPTION_IF_NULL(hash_table_ptr);
  return hash_table_ptr->Insert(static_cast<KeyType *>(inputs[kIndex1]->addr), inputs[kIndex1]->size / sizeof(KeyType),
                                static_cast<ValueType *>(inputs[kIndex2]->addr), stream_ptr);
}

bool MapTensorPutGpuKernelMod::InitSize(const BaseOperatorPtr &, const std::vector<KernelTensorPtr> &inputs,
                                        const std::vector<KernelTensorPtr> &outputs) {
  // Return size 1 as the first input size and the output size for MapTensorPut. Real map tensor is assigned by
  // framework.
  input_size_list_.push_back(kSizeOne);
  output_size_list_.push_back(kSizeOne);

  MS_EXCEPTION_IF_NULL(inputs[kIndex1]);
  auto put_key_size = inputs[kIndex1]->GetSizeInBytes();
  input_size_list_.push_back(put_key_size);

  MS_EXCEPTION_IF_NULL(inputs[kIndex2]);
  auto put_value_size = inputs[kIndex2]->GetSizeInBytes();
  input_size_list_.push_back(put_value_size);
  return true;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, MapTensorPut, MapTensorPutGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore