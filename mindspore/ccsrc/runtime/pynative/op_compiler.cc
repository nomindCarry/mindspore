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

#include "runtime/pynative/op_compiler.h"

#include <memory>
#include <algorithm>
#include <vector>
#include "backend/common/session/anf_runtime_algorithm.h"
#include "runtime/pynative/op_executor.h"
#include "runtime/pynative/op_runtime_info.h"
#include "runtime/device/device_address_utils.h"
#include "include/common/utils/scoped_long_running.h"

namespace mindspore {
using runtime::DeviceAddressUtils;
namespace pynative {
namespace {
void UpdateRefInfoBeforeCreateKernel(const session::BackendOpRunInfoPtr &op_run_info, const KernelGraphPtr &graph) {
  // Building Graph and Create Kernel is async, under pynative mode.Ref info is bind with kernel.
  // So need to get ref info to generate output addr, before create kernel.
  if (op_run_info->base_op_run_info.device_target != kCPUDevice &&
      op_run_info->base_op_run_info.device_target != kGPUDevice) {
    // just ascend ref mode is diff with cpu and gpu
    return;
  }

  AnfAlgo::AddOutInRefToGraph(graph);
}

void CreateDeviceAddressWithoutWorkspace(const KernelGraphPtr &graph, const DeviceContext *device_context,
                                         bool is_gradient_out) {
  DeviceAddressUtils::CreateParameterDeviceAddress(device_context, graph);
  DeviceAddressUtils::CreateValueNodeDeviceAddress(device_context, graph);
  DeviceAddressUtils::CreateKernelOutputDeviceAddress(device_context, graph, is_gradient_out);
  DeviceAddressUtils::UpdateDeviceAddressForInplaceNode(graph);
  DeviceAddressUtils::UpdateDeviceAddressForRefNode(graph);
}
}  // namespace

OpCompiler::OpCompiler() { session_ = session::SessionFactory::Get().Create(kSessionBasic); }

OpCompiler &OpCompiler::GetInstance() {
  static OpCompiler instance;
  return instance;
}

void OpCompiler::RunFinishAddPool(const OpCompilerInfoPtr &op_compiler_info, const GraphInfo &graph_info) {
  {
    std::unique_lock<std::mutex> lock(pool_mutex_);
    auto iter_pool = op_compiler_infos_pool_.find(graph_info);
    if (iter_pool != op_compiler_infos_pool_.end()) {
      iter_pool->second.emplace_back(op_compiler_info);
    } else {
      op_compiler_infos_pool_[graph_info] = {op_compiler_info};
    }
  }
}

OpCompilerInfoPtr OpCompiler::Compile(const session::BackendOpRunInfoPtr &op_run_info, bool *single_op_cache_hit,
                                      device::DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(device_context);
  py::gil_scoped_acquire acquire_gil;
  auto graph_info = op_run_info->base_op_run_info.graph_info;
  auto iter = op_compiler_infos_.find(graph_info);
  auto &op_executor = runtime::OpExecutor::GetInstance();
  if (iter != op_compiler_infos_.end() && op_executor.BuildInQueue(iter->second->graph_id_)) {
    op_executor.Wait();
  }
  {
    std::unique_lock<std::mutex> lock(pool_mutex_);
    auto iter_pool = op_compiler_infos_pool_.find(graph_info);
    // Check if the graph cache exists.
//  auto &op_executor = runtime::OpExecutor::GetInstance();
    if (iter_pool != op_compiler_infos_pool_.end()) {
      auto pool = iter_pool->second;
      if (!pool.empty()) {
        auto op_compiler_info_pool = *(iter_pool->second.end() - 1);
        iter_pool->second.pop_back();
        *single_op_cache_hit = true;
        return op_compiler_info_pool;
      }
    }
  }
  *single_op_cache_hit = false;
  // Generate kernel graph.
  MS_EXCEPTION_IF_NULL(session_);
  KernelGraphPtr graph = session_->ConstructSingleOpGraph(
    op_run_info, op_run_info->base_op_run_info.input_tensor, op_run_info->base_op_run_info.input_mask,
    device_context->GetDeviceType() == device::DeviceType::kAscend);
  MS_EXCEPTION_IF_NULL(graph);

  graph->set_run_mode(device::RunMode::kKernelMode);
  graph->set_is_from_single_op(true);
  MS_EXCEPTION_IF_NULL(device_context->kernel_executor_);
  // session_ is SessionBasic, AscendUnifyMindIR has not been executed.
  auto deprecated_kernel_executor =
    dynamic_cast<device::DeprecatedKernelExecutor *>(device_context->kernel_executor_.get());
  if (deprecated_kernel_executor != nullptr) {
    deprecated_kernel_executor->UnifyMindIR(graph);
  } else {
    opt::CommonUnifyMindIR(graph);
  }

  opt::OpBackendCommonOptimization(graph);

  // Select kernel and optimize
  device_context->kernel_executor_->OptimizeGraph(graph);

  UpdateRefInfoBeforeCreateKernel(op_run_info, graph);

  // Create device address for all anf nodes of graph.
  CreateDeviceAddressWithoutWorkspace(graph, device_context, op_run_info->is_gradient_out);

  auto output_nodes = graph->outputs();
  std::vector<KernelWithIndex> outputs_with_index;
  for (auto &node : output_nodes) {
    MS_EXCEPTION_IF_NULL(node);
    (void)outputs_with_index.emplace_back(common::AnfAlgo::VisitKernel(node, 0));
  }
  AnfAlgo::UpdateGraphValidRefPair(graph);

  auto op_compiler_info =
    std::make_shared<OpCompilerInfo>(graph_info, graph->graph_id(), graph, outputs_with_index, device_context, false);

  op_compiler_infos_[graph_info] = op_compiler_info;
//  op_compiler_infos_pool_[graph_info].emplace_back(op_compiler_info);
  return op_compiler_info;
}

void OpCompiler::BatchBuild(const std::vector<KernelGraphPtr> &graphs, const DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(device_context);
  std::vector<CNodePtr> node_to_build;
  for (const auto &graph : graphs) {
    const auto &nodes = graph->execution_order();
    (void)std::copy(nodes.begin(), nodes.end(), std::back_inserter(node_to_build));
  }
  // Kernel build
  device_context->kernel_executor_->CreateKernel(node_to_build);

  for (const auto &graph : graphs) {
    device_context->kernel_executor_->PreprocessBeforeRun(graph);
    DeviceAddressUtils::CreateKernelWorkspaceDeviceAddress(device_context, graph);
    // Need to execute after PreprocessBeforeRunSingleOpGraph
    runtime::OpRuntimeInfo::CacheGraphOpRuntimeInfo(graph);
  }
}

void OpCompiler::ClearOpCache(const GraphInfo &graph_info) {
  (void)op_compiler_infos_.erase(graph_info);
  {
    std::unique_lock<std::mutex> lock(pool_mutex_);
    (void)op_compiler_infos_pool_.erase(graph_info);
  }
}

void OpCompiler::ClearAllCache() { op_compiler_infos_.clear();
  {
    std::unique_lock<std::mutex> lock(pool_mutex_);
    op_compiler_infos_pool_.clear();
  }
}
}  // namespace pynative
}  // namespace mindspore
