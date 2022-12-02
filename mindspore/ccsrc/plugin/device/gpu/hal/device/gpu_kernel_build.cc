/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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
#include "plugin/device/gpu/hal/device/gpu_kernel_build.h"
#include <string>
#include <memory>
#include "kernel/kernel.h"
#ifndef _MSC_VER
#include "plugin/device/gpu/kernel/akg/akg_gpu_kernel_build.h"
#endif
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "kernel/common_utils.h"
#include "frontend/operator/ops.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "backend/common/session/kernel_build_client.h"
#include "plugin/device/gpu/hal/device/cuda_env_checker.h"
namespace mindspore {
namespace device {
namespace gpu {
namespace {
void SetGpuRefMapToKernelInfo(const CNodePtr &apply_kernel, const std::vector<kernel::KernelAttr> &kernel_attrs) {
  MS_EXCEPTION_IF_NULL(apply_kernel);
  if (kernel_attrs.empty()) {
    return;
  }

  auto kernel_attr = kernel::GetKernelAttrFromNode(apply_kernel);
  auto [is_match, index] = kernel::MatchKernelAttr(kernel_attr, kernel_attrs);
  if (kernel_attrs[0].GetSkipCheck()) {
    is_match = true;
    index = 0;
  }
  if (!is_match) {
    MS_LOG(EXCEPTION) << common::AnfAlgo::GetCNodeName(apply_kernel)
                      << " does not support this kernel data type: " << kernel_attr;
  }

  auto kernel_info = dynamic_cast<device::KernelInfo *>(apply_kernel->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  const auto &matched_kernel_attr = kernel_attrs[index];
  if (!matched_kernel_attr.GetOutInRefMap().empty() || matched_kernel_attr.GetAllOutInRef()) {
    kernel_info->set_ref_map(matched_kernel_attr.GetAllOutInRef(), matched_kernel_attr.GetOutInRefMap());
  }
}
}  // namespace

void InitAndResizeWithoutParameterInput(const CNodePtr &kernel,
                                        std::shared_ptr<kernel::NativeGpuKernelMod> gpu_kernel_mod,
                                        kernel::KernelArgs args,
                                        std::map<uint32_t, tensor::TensorPtr> inputs_tensor_map) {
  MS_EXCEPTION_IF_NULL(gpu_kernel_mod);
  if (!gpu_kernel_mod->Init(args.op, args.inputs, args.outputs)) {
    MS_LOG(EXCEPTION) << "Initialize gpu kernel op[" << kernel->fullname_with_scope() << "] failed.";
  }
  MS_EXCEPTION_IF_NULL(kernel);
  MS_EXCEPTION_IF_NULL(kernel->input(0));
  if (!AnfAlgo::NodeValueIsFuncGraph(kernel->input(0))) {
    const auto &depend_list = abstract::GetValueDependArgIndices(kernel);
    if (!depend_list.empty()) {
      auto input_size = common::AnfAlgo::GetInputTensorNum(kernel);
      for (size_t i = 0; i < input_size; ++i) {
        if (depend_list.find(i) == depend_list.end()) {
          continue;
        }
        auto input_node_with_index = common::AnfAlgo::GetPrevNodeOutput(kernel, i, false);
        auto real_input = input_node_with_index.first;
        // Inverse op have constant input need RunGraphBySingleOp
        if (real_input->isa<Parameter>()) {
          MS_LOG(DEBUG) << "Set Node Attr is Dynamic Shape";
          common::AnfAlgo::SetNodeAttr(mindspore::kAttrOutputIsDynamicShape, MakeValue(true), kernel);
          kernel->func_graph()->cast<KernelGraphPtr>()->SetGraphDynamicAttr(true);
          return;
        }
      }
    }
  }
  if (gpu_kernel_mod->Resize(args.op, args.inputs, args.outputs, inputs_tensor_map) == kernel::KRET_RESIZE_FAILED) {
    MS_LOG(EXCEPTION) << "gpu kernel op[" << kernel->fullname_with_scope() << "] Resize failed.";
  }
}
void CreateGPUKernel(const std::vector<CNodePtr> &kernels) {
  kernel::KernelMeta *bin_map = kernel::KernelMeta::GetInstance();
  MS_EXCEPTION_IF_NULL(bin_map);
  bool already_check_nvcc = false;
  std::vector<AnfNodePtr> akg_nodes;
  for (const auto &kernel : kernels) {
    MS_EXCEPTION_IF_NULL(kernel);
    std::string kernel_name = common::AnfAlgo::GetCNodeName(kernel);
    if (kernel_name == prim::kPrimTupleGetItem->name() || kernel_name == prim::kPrimMakeTuple->name() ||
        kernel_name == prim::kPrimDepend->name() || kernel_name == prim::kPrimStateSetItem->name()) {
      continue;
    }

    if (session::AnfRuntimeAlgorithm::GetKernelType(kernel) == KernelType::AKG_KERNEL) {
      if (!bin_map->initialized()) {
        bin_map->Initialize();
      }
      if (!already_check_nvcc) {
        already_check_nvcc = true;
        if (!CudaEnvChecker::GetInstance().CheckNvccInPath()) {
          MS_LOG(EXCEPTION)
            << "Failed to find nvcc compiler, please add nvcc position to the PATH environment variable, run "
               "the command: export PATH=${CUDA_PATH}/bin:${PATH}, CUDA_PATH is the installation path of the "
               "cuda library(eg. /usr/local/cuda).";
        }
      }
      akg_nodes.push_back(kernel);
    } else if (!common::AnfAlgo::IsControlOpExecInBackend(kernel)) {
      std::shared_ptr<kernel::NativeGpuKernelMod> gpu_kernel_mod = nullptr;
      bool new_factory = true;
      if (kernel::Factory<kernel::NativeGpuKernelMod>::Instance().IsRegistered(kernel_name)) {
        gpu_kernel_mod = kernel::Factory<kernel::NativeGpuKernelMod>::Instance().Create(kernel_name);
      } else {
        gpu_kernel_mod =
          (std::shared_ptr<kernel::NativeGpuKernelMod>)(kernel::NativeGpuKernelModFactory::GetInstance().Create(
            kernel_name, kernel));
        new_factory = false;
      }
      if (!gpu_kernel_mod) {
        MS_LOG(EXCEPTION) << "Build gpu kernel op[" << kernel->fullname_with_scope() << "] failed";
      }
      MS_EXCEPTION_IF_NULL(kernel);

      auto old_gpu_kernel_mod = std::dynamic_pointer_cast<kernel::DeprecatedNativeGpuKernelMod>(gpu_kernel_mod);
      auto args = kernel::AbstractArgsFromCNode(kernel, old_gpu_kernel_mod != nullptr);
      // inputs_tensor_map is ops's valueDepend input. if this input is const_value tensor,
      // we will put this tensor in args.inputs.host_data_.
      auto inputs_tensor_map = std::map<uint32_t, tensor::TensorPtr>();
      kernel::SetInputsByConstInputs(kernel, &inputs_tensor_map);
      kernel::SetInputsByDependMap(inputs_tensor_map, &args.inputs);
      if (old_gpu_kernel_mod) {
        kernel::SetArgsToCNode(kernel, args);
        if (new_factory) {
          old_gpu_kernel_mod->SetGpuRefMapToKernelInfo(kernel);
        }
        if (!old_gpu_kernel_mod->Init(kernel)) {
          MS_LOG(EXCEPTION) << "Initialize gpu kernel op[" << kernel->fullname_with_scope() << "] failed.";
        }
        session::AnfRuntimeAlgorithm::SetKernelMod(old_gpu_kernel_mod, kernel.get());
      } else {
        if (new_factory) {
          auto kernel_attrs = gpu_kernel_mod->GetOpSupport();
          SetGpuRefMapToKernelInfo(kernel, kernel_attrs);
        }
        auto ms_context = MsContext::GetInstance();
        MS_EXCEPTION_IF_NULL(ms_context);
        auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
        gpu_kernel_mod->SetDevicedId(device_id);

        InitAndResizeWithoutParameterInput(kernel, gpu_kernel_mod, args, inputs_tensor_map);

        session::AnfRuntimeAlgorithm::SetKernelMod(gpu_kernel_mod, kernel.get());
      }
    }
  }
#ifndef _MSC_VER
  kernel::AkgGpuKernelBuilder akg_gpu_kernel_builder;
  (void)akg_gpu_kernel_builder.AkgKernelParallelBuild(akg_nodes);
#endif
}
}  // namespace gpu
}  // namespace device
}  // namespace mindspore
