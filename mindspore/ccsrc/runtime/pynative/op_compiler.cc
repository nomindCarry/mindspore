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
#include "include/backend/anf_runtime_algorithm.h"
#include "runtime/pynative/op_executor.h"
#include "runtime/pynative/op_runtime_info.h"
#include "runtime/device/device_address_utils.h"
#include "backend/common/optimizer/common_backend_optimization.h"

namespace mindspore {
using runtime::DeviceAddressUtils;
namespace pynative {
namespace {
constexpr size_t kGraphInfoReserveLen = 128;

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

device::DeviceAddressPtr GetGraphMapToCacheAddress(
  const std::map<device::DeviceAddressPtr, device::DeviceAddressPtr> &graph_map_to_cache,
  const device::DeviceAddressPtr &device_address) {
  auto iter = graph_map_to_cache.find(device_address);
  if (iter != graph_map_to_cache.end()) {
    return iter->second;
  }
  return nullptr;
}

void CacheForGraphInputs(const OpCompilerInfoPtr &op_compiler_info,
                         std::map<device::DeviceAddressPtr, device::DeviceAddressPtr> *graph_map_to_cache) {
  auto &graph = op_compiler_info->graph_;
  MS_EXCEPTION_IF_NULL(graph);
  auto device_context = op_compiler_info->device_context_;
  const auto &inputs = graph->inputs();
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto &input = inputs[i];
    MS_EXCEPTION_IF_NULL(input);
    auto node_address = AnfAlgo::GetMutableOutputAddr(input, 0);
    MS_EXCEPTION_IF_NULL(node_address);
    auto cached_address = GetGraphMapToCacheAddress(*graph_map_to_cache, node_address);
    if (cached_address == nullptr) {
      cached_address = runtime::DeviceAddressUtils::CloneEmptyDeviceAddress(node_address, device_context);
      (*graph_map_to_cache)[node_address] = cached_address;
    }
    op_compiler_info->inputs_.emplace_back(cached_address);
  }
}

void CacheForGraphOutputs(const OpCompilerInfoPtr &op_compiler_info,
                          std::map<device::DeviceAddressPtr, device::DeviceAddressPtr> *graph_map_to_cache) {
  auto &graph = op_compiler_info->graph_;
  MS_EXCEPTION_IF_NULL(graph);
  auto device_context = op_compiler_info->device_context_;
  const auto &output_nodes = op_compiler_info->graph_output_nodes_;
  for (auto &item_with_index : output_nodes) {
    MS_EXCEPTION_IF_NULL(item_with_index.first);
    if (AnfAlgo::GetOutputTensorNum(item_with_index.first) == 0) {
      continue;
    }
    auto node_address = AnfAlgo::GetMutableOutputAddr(item_with_index.first, item_with_index.second, false);
    auto cached_address = GetGraphMapToCacheAddress(*graph_map_to_cache, node_address);
    if (cached_address == nullptr) {
      cached_address = runtime::DeviceAddressUtils::CloneEmptyDeviceAddress(node_address, device_context);
      (*graph_map_to_cache)[node_address] = cached_address;
    }
    op_compiler_info->outputs_.emplace_back(cached_address);
  }
}

void CacheForGraphValueNodes(const OpCompilerInfoPtr &op_compiler_info,
                             std::map<device::DeviceAddressPtr, device::DeviceAddressPtr> *graph_map_to_cache) {
  auto &graph = op_compiler_info->graph_;
  MS_EXCEPTION_IF_NULL(graph);
  const auto &value_nodes = graph->graph_value_nodes();
  for (auto &value_node : value_nodes) {
    if (!AnfAlgo::OutputAddrExist(value_node, 0, false)) {
      continue;
    }
    auto node_address = AnfAlgo::GetMutableOutputAddr(value_node, 0, false);
    (*graph_map_to_cache)[node_address] = node_address;

    const auto &node_value = value_node->value();
    MS_EXCEPTION_IF_NULL(node_value);
    if (node_value->isa<tensor::Tensor>()) {
      auto tensor = node_value->cast<tensor::TensorPtr>();
      MS_EXCEPTION_IF_NULL(tensor);
      op_compiler_info->value_map_to_tensor_[node_address] = tensor;
    } else {
      op_compiler_info->value_map_to_tensor_[node_address] = nullptr;
    }
  }
}

void CacheForGraphExecuteList(const OpCompilerInfoPtr &op_compiler_info,
                              std::map<device::DeviceAddressPtr, device::DeviceAddressPtr> *graph_map_to_cache) {
  MS_EXCEPTION_IF_NULL(op_compiler_info);
  auto &graph = op_compiler_info->graph_;
  MS_EXCEPTION_IF_NULL(graph);
  auto device_context = op_compiler_info->device_context_;
  const auto &nodes = graph->execution_order();
  for (auto const &node : nodes) {
    ExecuteKernelInfo exe_kernel_info;
    exe_kernel_info.kernel_ = node;

    // Save inputs
    auto input_num = common::AnfAlgo::GetInputTensorNum(node);
    for (size_t i = 0; i < input_num; ++i) {
      if (common::AnfAlgo::IsNoneInput(node, i)) {
        exe_kernel_info.inputs_device_address_.emplace_back(nullptr);
        continue;
      }
      session::KernelWithIndex kernel_with_index = common::AnfAlgo::GetPrevNodeOutput(node, i, false);
      auto node_address = AnfAlgo::GetMutableOutputAddr(kernel_with_index.first, kernel_with_index.second, false);
      auto cached_address = GetGraphMapToCacheAddress(*graph_map_to_cache, node_address);
      if (cached_address == nullptr) {
        cached_address = runtime::DeviceAddressUtils::CloneEmptyDeviceAddress(node_address, device_context);
        (*graph_map_to_cache)[node_address] = cached_address;
      }
      exe_kernel_info.inputs_device_address_.emplace_back(cached_address);
    }

    // Save outputs
    auto output_num = AnfAlgo::GetOutputTensorNum(node);
    for (size_t i = 0; i < output_num; ++i) {
      auto node_address = AnfAlgo::GetMutableOutputAddr(node, i, false);
      auto cached_address = GetGraphMapToCacheAddress(*graph_map_to_cache, node_address);
      if (cached_address == nullptr) {
        cached_address = runtime::DeviceAddressUtils::CloneEmptyDeviceAddress(node_address, device_context);
        (*graph_map_to_cache)[node_address] = cached_address;
      }
      exe_kernel_info.outputs_device_address_.emplace_back(cached_address);
    }

    op_compiler_info->execute_kernel_list_.emplace_back(exe_kernel_info);
  }
}
}  // namespace

OpCompiler::OpCompiler() { session_ = session::SessionFactory::Get().Create(kSessionBasic); }

OpCompiler &OpCompiler::GetInstance() {
  static OpCompiler instance;
  return instance;
}

KernelGraphPtr OpCompiler::GenerateKernelGraph(const session::BackendOpRunInfoPtr &op_run_info,
                                               device::DeviceContext *device_context) const {
  MS_EXCEPTION_IF_NULL(session_);
  KernelGraphPtr graph;
  if (op_run_info->op_prim->name() == "PackFunc") {
    auto recent_graph = op_run_info->op_prim->GetAttr("recent_graph");
    MS_EXCEPTION_IF_NULL(recent_graph);
    auto func_graph = recent_graph->cast<FuncGraphPtr>();
    std::vector<KernelGraphPtr> all_out_graph;
    graph = session_->ConstructKernelGraph(func_graph, &all_out_graph, device_context->GetDeviceType());
    graph->set_attr(kAttrPackFunction, MakeValue(True));
  } else {
    graph = session_->ConstructSingleOpGraph(op_run_info, op_run_info->base_op_run_info.input_tensor,
                                             op_run_info->base_op_run_info.input_mask,
                                             device_context->GetDeviceType() == device::DeviceType::kAscend);
  }
  return graph;
}

void OpCompiler::ConvertGraphToExecuteInfo(const OpCompilerInfoPtr &op_compiler_info) {
  MS_LOG(DEBUG) << "ConvertGraphToExecuteInfo";
  MS_EXCEPTION_IF_NULL(op_compiler_info);
  op_compiler_info->inputs_.clear();
  op_compiler_info->outputs_.clear();
  op_compiler_info->execute_kernel_list_.clear();

  std::map<device::DeviceAddressPtr, device::DeviceAddressPtr> graph_map_to_cache;

  // Save all value nodes
  CacheForGraphValueNodes(op_compiler_info, &graph_map_to_cache);

  // Save all inputs
  CacheForGraphInputs(op_compiler_info, &graph_map_to_cache);

  // Save all outputs
  CacheForGraphOutputs(op_compiler_info, &graph_map_to_cache);

  // Save all kernels
  CacheForGraphExecuteList(op_compiler_info, &graph_map_to_cache);
}

OpCompilerInfoPtr OpCompiler::Compile(const session::BackendOpRunInfoPtr &op_run_info, bool *single_op_cache_hit,
                                      device::DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(device_context);
  const auto &graph_info = op_run_info->base_op_run_info.graph_info;
  const auto &iter = op_compiler_infos_.find(graph_info);
  // Check if the graph cache exists.
  auto &op_executor = runtime::OpExecutor::GetInstance();
  if (iter != op_compiler_infos_.end() && op_executor.BuildQueueEmpty()) {
    const auto &op_compiler_info = iter->second;
    MS_EXCEPTION_IF_NULL(op_compiler_info);
    *single_op_cache_hit = true;
    return iter->second;
  }
  *single_op_cache_hit = false;
  // Generate kernel graph.
  MS_EXCEPTION_IF_NULL(session_);
  py::gil_scoped_acquire acquire_gil;
  KernelGraphPtr graph = GenerateKernelGraph(op_run_info, device_context);
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

  graph->set_graph_info(graph_info);
  ConvertGraphToExecuteInfo(op_compiler_info);
  op_compiler_infos_[graph_info] = op_compiler_info;
  return op_compiler_info;
}

void OpCompiler::BatchBuild(const std::vector<KernelGraphPtr> &graphs, const DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(device_context);
  // The compilation task may be in a child thread that has not yet set rt_context,
  // but the AICPU.so loading needs to use rt_context
  if (!device_context->device_res_manager_->BindDeviceToCurrentThread(true)) {
    MS_LOG(EXCEPTION) << "Bind device failed";
  }
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

std::string OpCompiler::GetSingleOpGraphInfo(const pynative::BaseOpRunInfo &op_info, const PrimitivePtr &op_prim) {
  MS_EXCEPTION_IF_NULL(op_prim);
  if (op_info.input_tensor.size() != op_info.input_mask.size()) {
    MS_LOG(EXCEPTION) << "Input tensors size " << op_info.input_tensor.size()
                      << " should be equal to tensors mask size " << op_info.input_mask.size();
  }
  std::string graph_info = op_info.device_target;
  graph_info.reserve(kGraphInfoReserveLen);

  if (op_info.use_dynamic_shape_process) {
    graph_info += "_1_";
  } else {
    graph_info += "_0_";
  }
  graph_info.append(op_prim->name()).append("_");
  bool has_hidden_side_effect = op_prim->HasAttr(GRAPH_FLAG_SIDE_EFFECT_HIDDEN);
  for (size_t index = 0; index < op_info.input_tensor.size(); ++index) {
    const auto &input_tensor = op_info.input_tensor[index];
    MS_EXCEPTION_IF_NULL(input_tensor);
    if (op_info.use_dynamic_shape_process) {
      graph_info += std::to_string(input_tensor->shape().size());
    } else {
      if (input_tensor->base_shape_ptr() != nullptr) {
        graph_info += input_tensor->base_shape_ptr()->ToString();
      } else {
        if (!input_tensor->shape().empty()) {
          const auto &shape_str =
            std::accumulate(std::next(input_tensor->shape().begin()), input_tensor->shape().end(),
                            std::to_string(input_tensor->shape()[0]),
                            [](std::string cur, size_t n) { return cur.append("-").append(std::to_string(n)); });
          graph_info += shape_str;
        }
      }
    }
    graph_info += std::to_string(input_tensor->data_type());
    graph_info += input_tensor->padding_type();
    // In the case of the same shape, but dtype and format are inconsistent
    auto tensor_addr = input_tensor->device_address();
    if (tensor_addr != nullptr && !has_hidden_side_effect) {
      auto p_address = std::dynamic_pointer_cast<device::DeviceAddress>(tensor_addr);
      MS_EXCEPTION_IF_NULL(p_address);
      graph_info += std::to_string(p_address->type_id());
      graph_info += p_address->format();
    }
    // For constant input
    if (op_info.input_mask[index] == kValueNodeTensorMask) {
      graph_info += common::AnfAlgo::GetTensorValueString(input_tensor);
    }
    graph_info += "_";
  }

  // Operator with hidden side effect.
  if (has_hidden_side_effect) {
    graph_info.append(std::to_string(op_prim->id())).append("_");
  }
  // The value of the attribute affects the operator selection
  const auto &attr_map = op_prim->attrs();
  (void)std::for_each(attr_map.begin(), attr_map.end(), [&graph_info](const auto &element) {
    if (element.first == kAttrInputNames || element.first == kAttrOutputNames) {
      return;
    }
    graph_info.append(element.second->ToString());
  });

  return graph_info;
}

void OpCompiler::ClearOpCache(const GraphInfo &graph_info) { (void)op_compiler_infos_.erase(graph_info); }

void OpCompiler::ClearAllCache() { op_compiler_infos_.clear(); }
}  // namespace pynative
}  // namespace mindspore
