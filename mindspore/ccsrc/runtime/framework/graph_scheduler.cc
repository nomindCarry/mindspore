/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "runtime/framework/graph_scheduler.h"
#include "runtime/framework/actor/memory_manager_actor.h"
#include "mindrt/src/actor/actormgr.h"
#include "mindrt/include/async/async.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "backend/optimizer/common/helper.h"
#include "utils/config_manager.h"
#include "utils/log_adapter.h"
#include "utils/convert_utils.h"
#include "utils/ms_context.h"
#include "common/trans.h"

namespace mindspore {
namespace runtime {
namespace {
bool IsDeviceQueueDSActor(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (node->isa<CNode>() && (AnfAlgo::GetCNodeName(node) == kGetNextOpName)) {
    return true;
  }
  return false;
}

bool IsHostQueueDSActor(const AnfNodePtr &node, const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(graph);
  if (node->isa<Parameter>() && (!AnfAlgo::IsParameterWeight(node->cast<ParameterPtr>()))) {
    //  Judge whether node is internal parameter.
    if (graph->GetFrontNodeByInternalParameter(node) == nullptr) {
      return true;
    }
  }
  return false;
}

bool IsKernelActor(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (node->isa<CNode>() && (AnfAlgo::GetCNodeName(node) != kGetNextOpName)) {
    return true;
  }
  return false;
}

// Judge whether the device tensor of the node is persistent or not.
bool IsPersistentDeviceTensor(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (node->isa<ValueNode>()) {
    return true;
  }
  if (node->isa<Parameter>() && AnfAlgo::IsParameterWeight(node->cast<ParameterPtr>())) {
    return true;
  }
  return false;
}

KernelActor *FindKernelActor(const KernelMapActor &kernel_actors_map, const std::string &name) {
  auto iter = kernel_actors_map.find(name);
  if (iter != kernel_actors_map.end()) {
    return iter->second.get();
  }
  return nullptr;
}

DeviceQueueDataSourceActor *FindDeviceQueueDSActor(const std::vector<DataSourceActorPtr> &data_source_actors) {
  for (auto &actor : data_source_actors) {
    MS_EXCEPTION_IF_NULL(actor);
    if (actor->GetAID().Name().find("_DeviceQueueDataSourceActor") != string::npos) {
      auto device_queue_ds_actor = dynamic_cast<DeviceQueueDataSourceActor *>(actor.get());
      return device_queue_ds_actor;
    }
  }
  return nullptr;
}

HostQueueDataSourceActor *FindHostQueueDSActor(const std::vector<DataSourceActorPtr> &data_source_actors) {
  for (auto &actor : data_source_actors) {
    MS_EXCEPTION_IF_NULL(actor);
    if (actor->GetAID().Name().find("_HostQueueDataSourceActor") != string::npos) {
      auto device_queue_ds_actor = dynamic_cast<HostQueueDataSourceActor *>(actor.get());
      return device_queue_ds_actor;
    }
  }
  return nullptr;
}

// Update the reference count of device tensor by the output index of node.
void UpdateRefCount(const AnfNodePtr &node, size_t output_idx, bool is_max_ref_count = false) {
  MS_EXCEPTION_IF_NULL(node);
  auto device_tensor = AnfAlgo::GetMutableOutputAddr(node, output_idx);
  MS_EXCEPTION_IF_NULL(device_tensor);
  if (is_max_ref_count) {
    device_tensor->set_original_ref_count(SIZE_MAX);
  } else {
    device_tensor->IncreaseOriginalRefCount();
  }
  device_tensor->ResetRefCount();
}

//  The branch processing of PrepareDataForValueNode that value type is tensor.
void PrepareDataForValueNodeTensor(const ValueNodePtr &node, const ValuePtr &node_value,
                                   const DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(node_value);
  MS_EXCEPTION_IF_NULL(device_context);

  std::vector<TensorPtr> tensors;
  TensorValueToTensor(node_value, &tensors);

  for (size_t i = 0; i < tensors.size(); i++) {
    const auto &tensor = tensors[i];
    if (tensor == nullptr) {
      MS_LOG(WARNING) << "Tensor is null";
      return;
    }

    const auto &device_tensor = AnfAlgo::GetMutableOutputAddr(node, i);
    MS_EXCEPTION_IF_NULL(device_tensor);
    // If the ptr of device tensor is not nullptr, it indicates that the device data has been prepared.
    if (device_tensor->GetPtr() != nullptr) {
      return;
    }
    MS_LOG(INFO) << "Prepare device data for value node: " << node->fullname_with_scope() << ", output index: " << i;
    tensor->set_device_address(device_tensor);

    // Allocate device memory.
    if (!device_context->AllocateMemory(device_tensor.get(), device_tensor->GetSize())) {
      MS_LOG(EXCEPTION) << "Device memory isn't enough and alloc failed, node name: " << node->fullname_with_scope()
                        << ", alloc size: " << device_tensor->GetSize();
    }

    // Copy data from host tensor to device.
    if (!device_tensor->SyncHostToDevice(trans::GetRuntimePaddingShape(node, 0), LongToSize(tensor->data().nbytes()),
                                         tensor->data_type(), tensor->data_c())) {
      MS_LOG(EXCEPTION) << "SyncHostToDevice failed, node name: " << node->fullname_with_scope();
    }
  }
}

// Prepare the device data for persistent device tensor of value node.
void PrepareDataForValueNode(const ValueNodePtr &node, const DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(device_context);
  auto &node_value = node->value();
  MS_EXCEPTION_IF_NULL(node_value);

  if (node_value->isa<tensor::Tensor>() || node_value->isa<ValueTuple>()) {
    //  The branch processing that value type is tensor.
    PrepareDataForValueNodeTensor(node, node_value, device_context);
  } else if (node_value->isa<StringImm>()) {
    const auto &device_tensor = AnfAlgo::GetMutableOutputAddr(node, 0);
    MS_EXCEPTION_IF_NULL(device_tensor);
    // If the ptr of device tensor is not nullptr, it indicates that the device data has been prepared.
    if (device_tensor->GetPtr() != nullptr) {
      return;
    }
    MS_LOG(INFO) << "Prepare device data for value node: " << node->fullname_with_scope();

    // Allocate device memory.
    if (!device_context->AllocateMemory(device_tensor.get(), device_tensor->GetSize())) {
      MS_LOG(EXCEPTION) << "Device memory isn't enough and alloc failed, node name: " << node->fullname_with_scope()
                        << ", alloc size: " << device_tensor->GetSize();
    }

    // Copy data from value to device.
    auto value = GetValue<std::string>(node_value);
    size_t tensor_size = value.size();
    ShapeVector shape = {1, SizeToLong(tensor_size)};
    if (!device_tensor->SyncHostToDevice(shape, tensor_size, kNumberTypeUInt8, value.data())) {
      MS_LOG(EXCEPTION) << "SyncHostToDevice failed, node name: " << node->fullname_with_scope();
    }
  }
}

// Prepare the device data for persistent device tensor of weight node from host tensor.
void PrepareDataForWeightNode(const AnfNodePtr &node, const TensorPtr &tensor, const DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(tensor);
  const auto &device_tensor = AnfAlgo::GetMutableOutputAddr(node, 0);
  MS_EXCEPTION_IF_NULL(device_tensor);
  const auto &host_tensor_address = std::dynamic_pointer_cast<DeviceTensor>(tensor->device_address());
  // If the host tensor has the device address, it indicates that the device address of host tensor is new.
  if (host_tensor_address != nullptr) {
    if (host_tensor_address != device_tensor) {
      AnfAlgo::SetOutputAddr(host_tensor_address, 0, node.get());
      DeviceTensorStore::GetInstance().Insert(node.get(), host_tensor_address);
    }
    return;
  }

  // If the ptr of device tensor is not nullptr, it indicates that the device data has been prepared.
  if (device_tensor->GetPtr() != nullptr) {
    return;
  }
  MS_LOG(INFO) << "Prepare device data for weight node: " << node->fullname_with_scope();
  tensor->set_device_address(device_tensor);

  // Allocate device memory.
  if (!device_context->AllocateMemory(device_tensor.get(), device_tensor->GetSize())) {
    MS_LOG(EXCEPTION) << "Device memory isn't enough and alloc failed, node name: " << node->fullname_with_scope()
                      << ", alloc size: " << device_tensor->GetSize();
  }

  // Copy data from host tensor to device.
  if (!device_tensor->SyncHostToDevice(trans::GetRuntimePaddingShape(node, 0), LongToSize(tensor->data().nbytes()),
                                       tensor->data_type(), tensor->data_c())) {
    MS_LOG(EXCEPTION) << "SyncHostToDevice failed, node name: " << node->fullname_with_scope();
  }
}

void AllocateContinuousMemoryForInput(const AnfNodePtr &kernel, const DeviceContext *device_context,
                                      bool is_all_nop_node) {
  MS_EXCEPTION_IF_NULL(kernel);
  MS_EXCEPTION_IF_NULL(device_context);
  bool is_need_alloc_memory = false;
  size_t total_size = 0;
  std::vector<size_t> size_list;
  std::vector<DeviceTensorPtr> addr_list;

  const auto &kernel_mod = AnfAlgo::GetKernelMod(kernel);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  const auto &intput_sizes = kernel_mod->GetInputSizeList();
  for (size_t i = 0; i < intput_sizes.size(); ++i) {
    DeviceTensorPtr device_tensor;
    if (is_all_nop_node) {
      // Graph may be all nop nodes and not remove nop node, so this can not skip nop node.
      device_tensor = AnfAlgo::GetPrevNodeMutableOutputAddr(kernel, i, false);
    } else {
      device_tensor = AnfAlgo::GetPrevNodeMutableOutputAddr(kernel, i, true);
    }
    MS_EXCEPTION_IF_NULL(device_tensor);
    //  In the scene of communication op and computing op parallel multi stream, the input address of communication op
    //  can't be reused, so set the max reference count.
    device_tensor->set_original_ref_count(SIZE_MAX);
    device_tensor->ResetRefCount();

    if (device_tensor->GetPtr() == nullptr) {
      is_need_alloc_memory = true;
    }
    total_size += intput_sizes[i];
    size_list.emplace_back(intput_sizes[i]);
    addr_list.emplace_back(device_tensor);
  }

  if (is_need_alloc_memory) {
    auto ret = device_context->AllocateContinuousMemory(addr_list, total_size, size_list);
    if (!ret) {
      MS_LOG(EXCEPTION) << "Malloc device memory failed.";
    }
  }
}

void AllocateContinuousMemoryForOutput(const AnfNodePtr &kernel, const DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(kernel);
  MS_EXCEPTION_IF_NULL(device_context);
  bool is_need_alloc_memory = false;
  size_t total_size = 0;
  std::vector<size_t> size_list;
  std::vector<DeviceTensorPtr> addr_list;

  const auto &kernel_mod = AnfAlgo::GetKernelMod(kernel);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  const auto &output_sizes = kernel_mod->GetOutputSizeList();
  for (size_t i = 0; i < output_sizes.size(); ++i) {
    const auto &device_tensor = AnfAlgo::GetMutableOutputAddr(kernel, i, false);
    MS_EXCEPTION_IF_NULL(device_tensor);
    // One time application for continuous memory, so set the max reference count.
    device_tensor->set_original_ref_count(SIZE_MAX);
    device_tensor->ResetRefCount();

    if (device_tensor->GetPtr() == nullptr) {
      is_need_alloc_memory = true;
    }
    total_size += output_sizes[i];
    size_list.emplace_back(output_sizes[i]);
    addr_list.emplace_back(device_tensor);
  }

  if (is_need_alloc_memory) {
    auto ret = device_context->AllocateContinuousMemory(addr_list, total_size, size_list);
    if (!ret) {
      MS_LOG(EXCEPTION) << "Malloc device memory failed.";
    }
  }
}
}  // namespace

void GraphScheduler::Initialize() {
  if (init_) {
    return;
  }
  init_ = true;

  auto actorMgr = ActorMgr::GetActorMgrRef();
  MS_EXCEPTION_IF_NULL(actorMgr);

  // Create the thread pool of actor runtime.
  auto max_thread_num = GetMaxThreadNum();
  MS_LOG(INFO) << "Max available thread number: " << max_thread_num;
  actorMgr->Initialize(max_thread_num);

  // Create memory manager actor.
  auto memory_manager_actor = std::make_shared<MemoryManagerActor>();
  MS_EXCEPTION_IF_NULL(memory_manager_actor);
  memory_manager_aid_ = memory_manager_actor->GetAID();
  // Schedule memory manager actor, bind single thread to response to memory alloc and free quickly.
  auto base_actor = static_cast<ActorReference>(memory_manager_actor);
  (void)actorMgr->Spawn(base_actor, false);
}

ActorSet *GraphScheduler::Transform(const GraphCompilerInfo &graph_compiler_info, GraphExecutionStrategy strategy) {
  MS_LOG(INFO) << "Graph(" << graph_compiler_info.name_ << ") transforms actor begin.";
  if (graph_compiler_info.graphs_.size() == 0) {
    MS_LOG(EXCEPTION) << "The number of graphs is zero.";
  }
  if (graph_compiler_info.graphs_.size() != graph_compiler_info.device_contexts_.size()) {
    MS_LOG(EXCEPTION) << "The number of graphs is not equal to the number of device contexts.";
  }

  Initialize();

  PersistDeviceTensor(graph_compiler_info);
  const auto &actor_set = Build(graph_compiler_info);
  Link(actor_set.get(), graph_compiler_info, strategy);
  actors_.emplace(actor_set->name_, actor_set);

  DumpActor(actor_set.get());
  if (!CheckActorValid(actor_set.get())) {
    MS_LOG(EXCEPTION) << "The actor set of " << graph_compiler_info.name_ << " is invalid.";
  }
  MS_LOG(INFO) << "Graph(" << graph_compiler_info.name_ << ") transforms actor end.";
  return actor_set.get();
}

void GraphScheduler::Schedule(const ActorSet *actor_set) {
  MS_EXCEPTION_IF_NULL(actor_set);
  auto actorMgr = ActorMgr::GetActorMgrRef();
  MS_EXCEPTION_IF_NULL(actorMgr);

  // Schedule dats source actors.
  for (auto &data_source_actor : actor_set->data_source_actors_) {
    MS_EXCEPTION_IF_NULL(data_source_actor);
    auto base_actor = static_cast<ActorReference>(data_source_actor);
    (void)actorMgr->Spawn(base_actor);
  }

  // Schedule kernel actors.
  for (auto &kernel_actor : actor_set->kernel_actors_) {
    MS_EXCEPTION_IF_NULL(kernel_actor);
    auto base_actor = static_cast<ActorReference>(kernel_actor);
    (void)actorMgr->Spawn(base_actor);
  }

  // Schedule loop count actor.
  if (actor_set->loop_count_actor_ != nullptr) {
    auto base_actor = static_cast<ActorReference>(actor_set->loop_count_actor_);
    (void)actorMgr->Spawn(base_actor);
  }

  // Schedule output actor.
  if (actor_set->output_actor_ != nullptr) {
    auto base_actor = static_cast<ActorReference>(actor_set->output_actor_);
    (void)actorMgr->Spawn(base_actor);
  }
}

void GraphScheduler::PrepareRun(const ActorSet *actor_set, const GraphCompilerInfo &graph_compiler_info,
                                const std::vector<std::vector<TensorPtr>> &input_tensors) {
  MS_EXCEPTION_IF_NULL(actor_set);
  std::vector<TensorPtr> host_tensors;
  const auto &host_data_source_actor = FindHostQueueDSActor(actor_set->data_source_actors_);
  if (host_data_source_actor != nullptr) {
    host_tensors.resize(host_data_source_actor->data_nodes_.size());
  }

  for (size_t i = 0; i < graph_compiler_info.graphs_.size(); ++i) {
    const auto &graph = graph_compiler_info.graphs_[i];
    const auto &device_context = graph_compiler_info.device_contexts_[i];
    MS_EXCEPTION_IF_NULL(graph);

    // 1.Prepare the data of device tensor store(value nodes of graph).
    for (const auto &value_node : graph->graph_value_nodes()) {
      if (AnfAlgo::OutputAddrExist(value_node, 0)) {
        PrepareDataForValueNode(value_node, device_context);
      }
    }

    // 1.Prepare the data of device tensor store(weights of graph), and fill host tensors for non weighted parameters.
    const auto &input_nodes = graph->input_nodes();
    const auto &tensors = input_tensors[i];
    for (size_t j = 0; j < input_nodes.size(); ++j) {
      const auto &input_node = input_nodes[j];
      const auto &input_tensor = tensors[j];
      MS_EXCEPTION_IF_NULL(input_node);
      if (IsPersistentDeviceTensor(input_node)) {
        // Prepare the device data for weights.
        PrepareDataForWeightNode(input_node, input_tensor, device_context);
      } else if (IsHostQueueDSActor(input_node, graph)) {
        MS_EXCEPTION_IF_NULL(host_data_source_actor);
        // Fill the host tensors for non weighted parameters.
        const auto &iter = host_data_source_actor->data_node_position_map_.find(input_node);
        if (iter != host_data_source_actor->data_node_position_map_.end()) {
          host_tensors[iter->second] = input_tensor;
        }
      }
    }

    // 2.Prepare the continuous memory for communication kernel.
    for (const auto &kernel : graph->execution_order()) {
      if (AnfAlgo::IsCommunicationOp(kernel)) {
        AllocateContinuousMemoryForInput(kernel, device_context, graph->is_all_nop_node());
        AllocateContinuousMemoryForOutput(kernel, device_context);
      }
    }
  }

  // 3.Prepare the data of host tensor queue(non weighted parameters of graph).
  if (host_data_source_actor != nullptr) {
    const auto &host_tensor_queue = FetchHostQueue(actor_set->name_);
    MS_EXCEPTION_IF_NULL(host_tensor_queue);
    host_tensor_queue->PushData(host_tensors);
  }
}

bool GraphScheduler::Run(const ActorSet *actor_set, GraphExecutionStrategy strategy) {
  MS_EXCEPTION_IF_NULL(actor_set);
  // Construct OpContext.
  OpContext<DeviceTensor> op_context;
  uuids::uuid sequential_num;
  std::vector<Promise<int>> result(1);
  op_context.sequential_num_ = &sequential_num;
  op_context.results_ = &result;

  // Trigger no input kernel actor running.
  for (auto &no_input_kernel_actor : actor_set->no_input_kernel_actors_) {
    MS_EXCEPTION_IF_NULL(no_input_kernel_actor);
    Async(no_input_kernel_actor->GetAID(), &KernelActor::RunOpControl, nullptr, &op_context);
  }

  // Trigger data source actor running.
  for (auto &data_source_actor : actor_set->data_source_actors_) {
    MS_EXCEPTION_IF_NULL(data_source_actor);
    Async(data_source_actor->GetAID(), &DataSourceActor::FetchData, &op_context);
  }

  // Trigger kernel actor running in the step execution strategy.
  if (strategy == GraphExecutionStrategy::kStep) {
    for (auto &kernel_actor : actor_set->kernel_actors_) {
      MS_EXCEPTION_IF_NULL(kernel_actor);
      Async(kernel_actor->GetAID(), &KernelActor::RunOpControl, nullptr, &op_context);
    }
  }

  // Get the run result.
  auto result_future = result[0].GetFuture();
  result_future.Wait();
  if (!result_future.IsOK()) {
    return false;
  }

  // Sync device stream.
  const auto &first_kernel_actor = actor_set->kernel_actors_[0];
  MS_EXCEPTION_IF_NULL(first_kernel_actor);
  const auto &device_context = first_kernel_actor->device_context_;
  MS_EXCEPTION_IF_NULL(device_context);
  if (!device_context->SyncStream()) {
    MS_LOG(ERROR) << "Sync stream failed.";
    return false;
  }

  return true;
}

ActorSet *GraphScheduler::Fetch(const ActorInfo &actor_info) const {
  auto iter = actors_.find(actor_info);
  if (iter != actors_.end()) {
    return iter->second.get();
  } else {
    MS_LOG(ERROR) << "Can't find the actors map of " << actor_info;
    return nullptr;
  }
}

ActorSetPtr GraphScheduler::Build(const GraphCompilerInfo &graph_compiler_info) {
  auto actor_set = std::make_shared<ActorSet>(graph_compiler_info.name_);
  MS_EXCEPTION_IF_NULL(actor_set);

  auto host_queue = std::make_shared<HostTensorQueue>();
  actor_to_host_queue_.emplace(actor_set->name_, host_queue);
  actor_set->data_source_actors_ = BuildDataSourceActor(graph_compiler_info, host_queue);
  actor_set->kernel_actors_ = BuildKernelActor(graph_compiler_info);
  actor_set->loop_count_actor_ = BuildLoopCountActor(graph_compiler_info);
  actor_set->output_actor_ = BuildOutputActor(graph_compiler_info);

  return actor_set;
}

void GraphScheduler::Link(ActorSet *actor_set, const GraphCompilerInfo &graph_compiler_info,
                          GraphExecutionStrategy strategy) {
  MS_EXCEPTION_IF_NULL(actor_set);
  KernelMapActor kernel_actors_temp_map;
  for (auto &actor : actor_set->kernel_actors_) {
    MS_EXCEPTION_IF_NULL(actor);
    kernel_actors_temp_map.emplace(actor->GetAID().Name(), actor);
  }

  // Foreach the execution order to link the actors.
  for (const auto &graph : graph_compiler_info.graphs_) {
    MS_EXCEPTION_IF_NULL(graph);
    auto execution_order = graph->execution_order();
    for (auto &kernel : execution_order) {
      if (!IsKernelActor(kernel)) {
        continue;
      }
      auto kernel_actor = FindKernelActor(kernel_actors_temp_map, kernel->fullname_with_scope());

      for (size_t i = 0; i < AnfAlgo::GetInputNum(kernel); ++i) {
        auto input_node = AnfAlgo::GetInputNode(kernel, i);
        // Link the control arrows of kernel actor by the auto monad, the inputs include monad node.
        LinkControlArrowByAutoMonad(kernel_actor, input_node, kernel_actors_temp_map);
        if (HasAbstractMonad(input_node)) {
          continue;  // No data arrow for monad input.
        }

        KernelWithIndex from_kernel_with_output_idx = AnfAlgo::VisitKernelWithReturnType(input_node, 0, true);
        KernelWithIndex to_kernel_with_input_idx = std::make_pair(kernel, i);
        auto from_kernel = from_kernel_with_output_idx.first;

        if (IsDeviceQueueDSActor(from_kernel)) {
          // Link the data arrows of device queue data source actor.
          auto from_actor = FindDeviceQueueDSActor(actor_set->data_source_actors_);
          LinkDataArrowForDeviceDSActor(from_actor, kernel_actor, from_kernel_with_output_idx,
                                        to_kernel_with_input_idx);
        } else if (IsHostQueueDSActor(from_kernel, graph)) {
          // Link the data arrows of host queue data source actor.
          auto from_actor = FindHostQueueDSActor(actor_set->data_source_actors_);
          LinkDataArrowForHostDSActor(from_actor, kernel_actor, from_kernel_with_output_idx, to_kernel_with_input_idx);
        } else {
          // Link the data arrows of kernel actor.
          auto from_actor = FindKernelActor(kernel_actors_temp_map, from_kernel->fullname_with_scope());
          LinkDataArrowForKernelActor(from_actor, kernel_actor, from_kernel_with_output_idx, to_kernel_with_input_idx);
        }
      }
    }
  }

  // Link the control arrows of kernel actors.
  LinkControlArrowForKernelActor(&(actor_set->kernel_actors_), actor_set->loop_count_actor_.get(), strategy);

  // BuildNoInputKernelActor depends on whether kernel actors have input, so must be behind the link of kernel actors.
  auto no_input_kernel_actors = BuildNoInputKernelActor(actor_set);
  actor_set->no_input_kernel_actors_.swap(no_input_kernel_actors);

  // Link the control arrows of loop count actor, which depends on the no input kernel actors.
  LinkControlArrowForLoopCountActor(actor_set->loop_count_actor_.get(), actor_set);

  // Link the output result arrows for output actors.
  LinkOutputResultArrowForOutputActor(actor_set->output_actor_.get(), actor_set->data_source_actors_,
                                      kernel_actors_temp_map, graph_compiler_info);
}

std::vector<DataSourceActorPtr> GraphScheduler::BuildDataSourceActor(const GraphCompilerInfo &graph_compiler_info,
                                                                     const HostTensorQueuePtr &host_queue) {
  std::vector<DataSourceActorPtr> data_source_actors;
  HostQueueDSActorPtr host_queue_ds_actor = nullptr;
  size_t data_node_position = 0;
  std::unordered_map<AnfNodePtr, size_t> front_node_position_temp_map;

  for (size_t i = 0; i < graph_compiler_info.graphs_.size(); ++i) {
    const auto &graph = graph_compiler_info.graphs_[i];
    const auto &device_context = graph_compiler_info.device_contexts_[i];
    MS_EXCEPTION_IF_NULL(graph);
    // Build host queue data source actor.
    for (const auto &input_node : graph->input_nodes()) {
      MS_EXCEPTION_IF_NULL(input_node);
      if (IsHostQueueDSActor(input_node, graph)) {
        if (host_queue_ds_actor == nullptr) {
          auto actor_name = graph_compiler_info.name_ + "_HostQueueDataSourceActor";
          MS_LOG(INFO) << "Create host queue data source actor: " << actor_name;
          host_queue_ds_actor =
            std::make_shared<HostQueueDataSourceActor>(actor_name, 1, device_context, memory_manager_aid_, host_queue);
          data_source_actors.emplace_back(host_queue_ds_actor);
        }

        const auto &front_node = graph->GetFrontAnfByBackendAnf(input_node);
        // In the scenario where multiple backend nodes correspond to the same front node, only the first backend node
        // is saved in the host queue data source actor.
        if ((front_node != nullptr) && (front_node_position_temp_map.count(front_node) > 0)) {
          host_queue_ds_actor->data_node_position_map_.emplace(input_node, front_node_position_temp_map[front_node]);
          continue;
        }
        host_queue_ds_actor->data_nodes_.emplace_back(input_node);
        host_queue_ds_actor->data_node_position_map_.emplace(input_node, data_node_position);
        front_node_position_temp_map.emplace(front_node, data_node_position);
        data_node_position++;
      }
    }

    // Build device queue data source actor.
    const auto &execution_order = graph->execution_order();
    const auto &iter = std::find_if(execution_order.begin(), execution_order.end(),
                                    [](const CNodePtr &node) { return IsDeviceQueueDSActor(node); });
    if (iter != execution_order.end()) {
      auto actor_name =
        graph_compiler_info.name_ + "_DeviceQueueDataSourceActor" + "_" + std::to_string(graph->graph_id());
      MS_LOG(INFO) << "Create queue data source actor: " << actor_name;
      auto device_queue_ds_actor =
        std::make_shared<DeviceQueueDataSourceActor>(actor_name, 1, device_context, memory_manager_aid_);
      MS_EXCEPTION_IF_NULL(device_queue_ds_actor);
      data_source_actors.emplace_back(device_queue_ds_actor);
      device_queue_ds_actor->data_kernel_ = *iter;
    }
  }
  return data_source_actors;
}

std::vector<KernelActorPtr> GraphScheduler::BuildKernelActor(const GraphCompilerInfo &graph_compiler_info) {
  std::vector<KernelActorPtr> kernel_actors;

  for (size_t i = 0; i < graph_compiler_info.graphs_.size(); ++i) {
    const auto &graph = graph_compiler_info.graphs_[i];
    const auto &device_context = graph_compiler_info.device_contexts_[i];
    MS_EXCEPTION_IF_NULL(graph);
    auto execution_order = graph->execution_order();
    for (auto &kernel : execution_order) {
      if (IsKernelActor(kernel)) {
        auto kernel_actor =
          std::make_shared<KernelActor>(kernel->fullname_with_scope(), kernel, device_context, memory_manager_aid_);
        MS_EXCEPTION_IF_NULL(kernel_actor);
        kernel_actors.emplace_back(kernel_actor);
      }
    }
  }
  return kernel_actors;
}

LoopCountActorPtr GraphScheduler::BuildLoopCountActor(const GraphCompilerInfo &graph_compiler_info) {
  auto loop_count = ConfigManager::GetInstance().iter_num();
  auto actor_name = graph_compiler_info.name_ + "_" + "LoopCountActor";
  auto loop_count_actor = std::make_shared<LoopCountActor>(actor_name, loop_count);
  MS_LOG(INFO) << "Create loop count actor: " << actor_name;
  MS_EXCEPTION_IF_NULL(loop_count_actor);
  return loop_count_actor;
}

OutputActorPtr GraphScheduler::BuildOutputActor(const GraphCompilerInfo &graph_compiler_info) {
  auto loop_count = ConfigManager::GetInstance().iter_num();
  auto actor_name = graph_compiler_info.name_ + "_" + "OutputActor";
  auto output_actor =
    std::make_shared<OutputActor>(actor_name, loop_count, graph_compiler_info.origin_outputs_order_.size());
  MS_LOG(INFO) << "Create output actor: " << actor_name;
  MS_EXCEPTION_IF_NULL(output_actor);
  return output_actor;
}

std::vector<KernelActorPtr> GraphScheduler::BuildNoInputKernelActor(const ActorSet *actor_set) {
  MS_EXCEPTION_IF_NULL(actor_set);
  std::vector<KernelActorPtr> no_input_kernel_actors;

  for (auto &kernel_actor : actor_set->kernel_actors_) {
    MS_EXCEPTION_IF_NULL(kernel_actor);
    if ((kernel_actor->input_datas_num_ == 0) && (kernel_actor->input_controls_num_ == 0)) {
      no_input_kernel_actors.emplace_back(kernel_actor);
      // The no input kernel actor will be triggered by loop count actor, so need set the input_controls_num_.
      kernel_actor->input_controls_num_ = 1;
    }
  }
  return no_input_kernel_actors;
}

void GraphScheduler::LinkDataArrowForDeviceDSActor(DeviceQueueDataSourceActor *from_actor, KernelActor *to_actor,
                                                   KernelWithIndex from_kernel_with_output_idx,
                                                   KernelWithIndex to_kernel_with_input_idx) {
  MS_EXCEPTION_IF_NULL(from_actor);
  MS_EXCEPTION_IF_NULL(to_actor);

  auto from_kernel = from_kernel_with_output_idx.first;
  MS_EXCEPTION_IF_NULL(from_kernel);
  auto from_output_index = from_kernel_with_output_idx.second;
  auto to_input_index = to_kernel_with_input_idx.second;

  auto to_aid = to_actor->GetAID();
  auto op_arrow = std::make_shared<OpArrow>(from_output_index, to_aid, to_input_index);
  from_actor->output_op_arrows_.emplace_back(op_arrow);
  to_actor->input_datas_num_++;

  // Update the reference count of device tensor.
  UpdateRefCount(from_kernel, from_output_index);
}

void GraphScheduler::LinkDataArrowForHostDSActor(HostQueueDataSourceActor *from_actor, KernelActor *to_actor,
                                                 KernelWithIndex from_kernel_with_output_idx,
                                                 KernelWithIndex to_kernel_with_input_idx) {
  MS_EXCEPTION_IF_NULL(from_actor);
  MS_EXCEPTION_IF_NULL(to_actor);

  auto from_kernel = from_kernel_with_output_idx.first;
  MS_EXCEPTION_IF_NULL(from_kernel);
  auto from_output_index = from_kernel_with_output_idx.second;
  auto to_input_index = to_kernel_with_input_idx.second;

  // Get the position of from kernel in the data source actor.
  auto iter = from_actor->data_node_position_map_.find(from_kernel);
  if (iter == from_actor->data_node_position_map_.end()) {
    MS_LOG(EXCEPTION) << "Parameter node: " << from_kernel->fullname_with_scope() << " is not exist.";
  }
  auto position = iter->second;

  auto to_aid = to_actor->GetAID();
  auto op_arrow = std::make_shared<OpArrow>(position, to_aid, to_input_index);
  from_actor->output_op_arrows_.emplace_back(op_arrow);
  to_actor->input_datas_num_++;

  // Update the reference count of device tensor.
  UpdateRefCount(from_actor->data_nodes_[position], from_output_index);
}

void GraphScheduler::LinkDataArrowForKernelActor(KernelActor *from_actor, KernelActor *to_actor,
                                                 KernelWithIndex from_kernel_with_output_idx,
                                                 KernelWithIndex to_kernel_with_input_idx) {
  MS_EXCEPTION_IF_NULL(to_actor);
  auto from_kernel = from_kernel_with_output_idx.first;
  MS_EXCEPTION_IF_NULL(from_kernel);
  auto from_output_index = from_kernel_with_output_idx.second;
  auto to_input_index = to_kernel_with_input_idx.second;

  if (IsPersistentDeviceTensor(from_kernel)) {
    to_actor->device_tensor_store_keys_.emplace_back(to_input_index, static_cast<void *>(from_kernel.get()));
  } else if (IsKernelActor(from_kernel)) {
    MS_EXCEPTION_IF_NULL(from_actor);
    auto to_aid = to_actor->GetAID();
    auto op_arrow = std::make_shared<OpArrow>(from_output_index, to_aid, to_input_index);
    from_actor->output_op_arrows_.emplace_back(op_arrow);
    to_actor->input_datas_num_++;

    // Update the reference count of device tensor.
    UpdateRefCount(from_kernel, from_output_index);
  }
}

void GraphScheduler::LinkControlArrowForKernelActor(std::vector<KernelActorPtr> *from_actors, LoopCountActor *to_actor,
                                                    GraphExecutionStrategy strategy) {
  MS_EXCEPTION_IF_NULL(from_actors);
  MS_EXCEPTION_IF_NULL(to_actor);

  for (auto &from_actor : *from_actors) {
    MS_EXCEPTION_IF_NULL(from_actor);
    if (strategy == GraphExecutionStrategy::kStep) {
      from_actor->input_controls_num_++;
    }

    // If the kernel actor has no output in the pipeline mode, then adds the output control to loop count actor.
    if ((strategy == GraphExecutionStrategy::kPipeline) && (from_actor->output_op_arrows_.size() == 0) &&
        (from_actor->output_op_controls_.size() == 0)) {
      MS_EXCEPTION_IF_NULL(from_actor->kernel_);
      MS_LOG(INFO) << from_actor->kernel_->fullname_with_scope() << " is not real used by other nodes.";
      auto to_aid = to_actor->GetAID();
      from_actor->output_op_controls_.emplace_back(to_aid);
      to_actor->input_controls_num_++;
    }
  }
}

void GraphScheduler::LinkControlArrowByAutoMonad(KernelActor *to_actor, const AnfNodePtr &from_node,
                                                 const KernelMapActor &kernel_actors_map) {
  MS_EXCEPTION_IF_NULL(to_actor);
  MS_EXCEPTION_IF_NULL(from_node);
  if (!from_node->isa<CNode>()) {
    return;
  }
  // Find the real input node, include the monad node and make tuple node.
  const std::vector<PrimitivePtr> &return_types = {prim::kPrimUpdateState, prim::kPrimLoad, prim::kPrimMakeTuple};
  const auto &input_kernel_with_output_idx = AnfAlgo::VisitKernelWithReturnType(from_node, 0, true, return_types);
  MS_EXCEPTION_IF_NULL(input_kernel_with_output_idx.first);
  if (!input_kernel_with_output_idx.first->isa<CNode>()) {
    return;
  }
  const auto &input_cnode = input_kernel_with_output_idx.first->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(input_cnode);

  // Get the real depend input by monad node which needs to link the control arrow.
  AnfNodePtr real_depend_input = nullptr;
  if (AnfAlgo::CheckPrimitiveType(input_cnode, prim::kPrimUpdateState)) {
    real_depend_input = input_cnode->input(kUpdateStateRealInput);
  } else if (AnfAlgo::CheckPrimitiveType(input_cnode, prim::kPrimLoad)) {
    real_depend_input = input_cnode->input(kLoadStateInput);
  } else if (AnfAlgo::CheckPrimitiveType(input_cnode, prim::kPrimMakeTuple)) {
    // Make tuple node needs to be expanded.
    for (size_t i = 1; i < input_cnode->inputs().size(); ++i) {
      LinkControlArrowByAutoMonad(to_actor, input_cnode->input(i), kernel_actors_map);
    }
    return;
  } else {
    return;
  }

  MS_EXCEPTION_IF_NULL(real_depend_input);
  if (!real_depend_input->isa<CNode>()) {
    return;
  }
  // The monad node and make tuple node need recursion.
  if (AnfAlgo::CheckPrimitiveType(real_depend_input, prim::kPrimUpdateState) ||
      AnfAlgo::CheckPrimitiveType(real_depend_input, prim::kPrimLoad) ||
      AnfAlgo::CheckPrimitiveType(real_depend_input, prim::kPrimMakeTuple)) {
    LinkControlArrowByAutoMonad(to_actor, real_depend_input, kernel_actors_map);
    return;
  }

  // Link the control arrow between the kernel actors.
  auto from_actor = FindKernelActor(kernel_actors_map, real_depend_input->fullname_with_scope());
  MS_EXCEPTION_IF_NULL(from_actor);
  from_actor->output_op_controls_.emplace_back(to_actor->GetAID());
  to_actor->input_controls_num_++;
}

void GraphScheduler::LinkControlArrowForLoopCountActor(LoopCountActor *loop_count_actor, const ActorSet *actor_set) {
  MS_EXCEPTION_IF_NULL(actor_set);
  MS_EXCEPTION_IF_NULL(loop_count_actor);

  // Set the source data actor.
  for (auto &data_source_actor : actor_set->data_source_actors_) {
    MS_EXCEPTION_IF_NULL(data_source_actor);
    loop_count_actor->data_source_aids_.emplace_back(data_source_actor->GetAID());
  }

  // Set the no input kernel actor.
  for (auto &no_input_kernel_actor : actor_set->no_input_kernel_actors_) {
    MS_EXCEPTION_IF_NULL(no_input_kernel_actor);
    loop_count_actor->no_input_kernel_aids_.emplace_back(no_input_kernel_actor->GetAID());
  }

  // Set the output actor.
  MS_EXCEPTION_IF_NULL(actor_set->output_actor_);
  loop_count_actor->output_aid_ = actor_set->output_actor_->GetAID();
}

void GraphScheduler::LinkOutputResultArrowForOutputActor(OutputActor *to_actor,
                                                         const std::vector<DataSourceActorPtr> &data_source_actors,
                                                         const KernelMapActor &kernel_actors_map,
                                                         const GraphCompilerInfo &graph_compiler_info) {
  MS_EXCEPTION_IF_NULL(to_actor);

  size_t number = 0;
  for (const auto &graph : graph_compiler_info.graphs_) {
    MS_EXCEPTION_IF_NULL(graph);
    ++number;
    const auto &outputs = AnfAlgo::GetAllOutput(graph->output(), {prim::kPrimTupleGetItem});
    for (const auto &output : outputs) {
      const auto &output_with_index = AnfAlgo::VisitKernelWithReturnType(output, 0, true);
      MS_EXCEPTION_IF_NULL(output_with_index.first);
      const auto &front_node = graph->GetFrontAnfByBackendAnf(output_with_index.first);
      if (front_node == nullptr) {
        continue;
      }

      auto origin_output_with_index = KernelWithIndex(front_node, output_with_index.second);
      const auto &iter = graph_compiler_info.origin_outputs_order_.find(origin_output_with_index);
      if (iter == graph_compiler_info.origin_outputs_order_.end()) {
        continue;
      }
      to_actor->device_contexts_[iter->second] = graph_compiler_info.device_contexts_[number - 1];
      // The device tensor of graph out need be taken over by host tensor, so set the max reference count.
      UpdateRefCount(output_with_index.first, output_with_index.second, true);

      // The graph output is from device tensor store.
      if (IsPersistentDeviceTensor(output_with_index.first)) {
        to_actor->device_tensor_store_keys_.emplace_back(iter->second, output_with_index.first);
        continue;
      }

      // The graph output is from kernel actor.
      if (IsKernelActor(output_with_index.first)) {
        const auto &from_actor = FindKernelActor(kernel_actors_map, output_with_index.first->fullname_with_scope());
        MS_EXCEPTION_IF_NULL(from_actor);
        auto op_arrow = std::make_shared<OpArrow>(output_with_index.second, to_actor->GetAID(), iter->second);
        from_actor->output_result_arrows_.emplace_back(op_arrow);
        continue;
      }

      // The graph output is from data source actor.
      DataSourceActor *from_actor = nullptr;
      size_t from_actor_output_index = 0;
      if (IsHostQueueDSActor(output_with_index.first, graph)) {
        const auto &host_queue_ds_actor = FindHostQueueDSActor(data_source_actors);
        auto position_iter = host_queue_ds_actor->data_node_position_map_.find(output_with_index.first);
        if (position_iter == host_queue_ds_actor->data_node_position_map_.end()) {
          MS_LOG(EXCEPTION) << "Parameter node: " << output_with_index.first->fullname_with_scope() << " is not exist.";
        }
        from_actor_output_index = position_iter->second;
        UpdateRefCount(host_queue_ds_actor->data_nodes_[from_actor_output_index], output_with_index.second, true);
        from_actor = static_cast<DataSourceActor *>(host_queue_ds_actor);
      } else if (IsDeviceQueueDSActor(output_with_index.first)) {
        from_actor = FindDeviceQueueDSActor(data_source_actors);
        from_actor_output_index = output_with_index.second;
      }
      MS_EXCEPTION_IF_NULL(from_actor);
      auto op_arrow = std::make_shared<OpArrow>(from_actor_output_index, to_actor->GetAID(), iter->second);
      from_actor->output_result_arrows_.emplace_back(op_arrow);
    }
  }
}

bool GraphScheduler::CheckActorValid(const ActorSet *actor_set) const {
  MS_EXCEPTION_IF_NULL(actor_set);
  // Check the data source actors.
  for (const auto &data_source_actor : actor_set->data_source_actors_) {
    MS_EXCEPTION_IF_NULL(data_source_actor);
    if (data_source_actor->output_op_arrows_.size() == 0) {
      MS_LOG(ERROR) << data_source_actor->GetAID().Name() << " has no user.";
      return false;
    }
  }

  // Check the kernel actors.
  for (const auto &kernel_actor : actor_set->kernel_actors_) {
    MS_EXCEPTION_IF_NULL(kernel_actor);
    if (kernel_actor->output_op_arrows_.size() + kernel_actor->output_op_controls_.size() == 0) {
      MS_LOG(ERROR) << kernel_actor->GetAID().Name() << " has no user.";
      return false;
    }

    auto input_num = AnfAlgo::GetInputTensorNum(kernel_actor->kernel_);
    auto input_data_num = kernel_actor->input_datas_num_;
    auto device_tensor_store_num = kernel_actor->device_tensor_store_keys_.size();
    if (input_data_num + device_tensor_store_num != input_num) {
      MS_LOG(ERROR) << "The input building of " << kernel_actor->GetAID().Name()
                    << " is wrong, input data num: " << input_data_num
                    << ", device tensor store num: " << device_tensor_store_num << ", total input num: " << input_num;
      return false;
    }
  }

  // Check the loop count actor.
  const auto &loop_count_actor = actor_set->loop_count_actor_;
  if (loop_count_actor != nullptr) {
    if (loop_count_actor->input_controls_num_ == 0) {
      MS_LOG(ERROR) << loop_count_actor->GetAID().Name() << " has no source.";
      return false;
    }
  }

  return true;
}

void GraphScheduler::PersistDeviceTensor(const GraphCompilerInfo &graph_compiler_info) {
  for (const auto &graph : graph_compiler_info.graphs_) {
    MS_EXCEPTION_IF_NULL(graph);

    for (auto &value_node : graph->graph_value_nodes()) {
      MS_EXCEPTION_IF_NULL(value_node);
      if (!AnfAlgo::OutputAddrExist(value_node, 0)) {
        MS_LOG(INFO) << "The device address is not exist: " << value_node->ToString();
        continue;
      }
      auto device_tensor = AnfAlgo::GetMutableOutputAddr(value_node, 0);
      DeviceTensorStore::GetInstance().Insert(value_node.get(), device_tensor);
      device_tensor->set_original_ref_count(SIZE_MAX);
      device_tensor->ResetRefCount();
    }

    for (auto &input_node : graph->input_nodes()) {
      MS_EXCEPTION_IF_NULL(input_node);
      if (IsPersistentDeviceTensor(input_node)) {
        auto device_tensor = AnfAlgo::GetMutableOutputAddr(input_node, 0);
        MS_EXCEPTION_IF_NULL(device_tensor);
        DeviceTensorStore::GetInstance().Insert(input_node.get(), device_tensor);
        device_tensor->set_original_ref_count(SIZE_MAX);
        device_tensor->ResetRefCount();
      }
    }
  }
}

HostTensorQueue *GraphScheduler::FetchHostQueue(const ActorInfo &actor_info) const {
  const auto &iter = actor_to_host_queue_.find(actor_info);
  if (iter != actor_to_host_queue_.end()) {
    return iter->second.get();
  } else {
    return nullptr;
  }
}

void GraphScheduler::DumpActor(const ActorSet *actor_set) const {
  MS_EXCEPTION_IF_NULL(actor_set);
  const auto &context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  auto save_graphs = context_ptr->get_param<bool>(MS_CTX_SAVE_GRAPHS_FLAG);
  if (!save_graphs) {
    return;
  }
  auto save_graphs_path = context_ptr->get_param<std::string>(MS_CTX_SAVE_GRAPHS_PATH);
  if (save_graphs_path.empty()) {
    save_graphs_path = ".";
  }

  std::string filename = save_graphs_path + "/actor_set_" + actor_set->name_ + ".ir";
  std::ofstream ofs(filename);
  if (!ofs.is_open()) {
    MS_LOG(ERROR) << "Open file [" << filename << "] failed!";
    return;
  }

  ofs << "[Data source actors]\n";
  for (const auto &data_source_actor : actor_set->data_source_actors_) {
    DumpDSActor(data_source_actor.get(), ofs);
    ofs << "\n";
  }

  ofs << "\n[Kernel actors]\n";
  for (const auto &kernel_actor : actor_set->kernel_actors_) {
    DumpKernelActor(kernel_actor.get(), ofs);
    ofs << "\n";
  }

  ofs << "\n[No input kernel actors]\n";
  for (const auto &no_input_kernel_actor : actor_set->no_input_kernel_actors_) {
    DumpKernelActor(no_input_kernel_actor.get(), ofs);
    ofs << "\n";
  }

  ofs << "\n[Loop count actor]\n";
  const auto &loop_count_actor = actor_set->loop_count_actor_;
  if (loop_count_actor != nullptr) {
    DumpLoopCountActor(loop_count_actor.get(), ofs);
    ofs << "\n";
  }

  ofs << "\n[Output actor]\n";
  const auto &output_actor = actor_set->output_actor_;
  if (output_actor != nullptr) {
    DumpOutputActor(output_actor.get(), ofs);
    ofs << "\n";
  }
}

void GraphScheduler::DumpDSActor(const DataSourceActor *actor, std::ofstream &ofs) const {
  MS_EXCEPTION_IF_NULL(actor);
  const auto &actor_name = actor->GetAID().Name();

  MS_EXCEPTION_IF_NULL(actor->device_context_);
  ofs << "\tactor_name:" << actor_name << "\tdevice_context:" << actor->device_context_->device_context_key().ToString()
      << "\n";

  if (actor_name.find("_DeviceQueueDataSourceActor") != string::npos) {
    // Dump the member info of device queue data source actor.
    const auto &device_queue_ds_actor = dynamic_cast<const DeviceQueueDataSourceActor *>(actor);
    const auto &data_kernel = device_queue_ds_actor->data_kernel_;
    MS_EXCEPTION_IF_NULL(data_kernel);
    ofs << "\t\tdata_kernel_name:" << data_kernel->fullname_with_scope()
        << "\tinput_number:" << AnfAlgo::GetInputTensorNum(data_kernel)
        << "\toutput_number:" << AnfAlgo::GetOutputTensorNum(data_kernel) << "\n";
    for (size_t i = 0; i < AnfAlgo::GetOutputTensorNum(data_kernel); ++i) {
      const auto &device_tensor = AnfAlgo::GetMutableOutputAddr(data_kernel, i, false);
      MS_EXCEPTION_IF_NULL(device_tensor);
      ofs << "\t\t\toutput_index:" << i << "\tptr:" << device_tensor->GetPtr() << "\tsize:" << device_tensor->GetSize()
          << "\toriginal_ref_count:" << device_tensor->original_ref_count() << "\n ";
    }
  } else if (actor_name.find("_HostQueueDataSourceActor") != string::npos) {
    // Dump the member info of host queue data source actor.
    const auto &host_queue_ds_actor = dynamic_cast<const HostQueueDataSourceActor *>(actor);
    ofs << "\t\tdata_nodes:" << host_queue_ds_actor->data_nodes_.size() << "\n";
    for (size_t i = 0; i < host_queue_ds_actor->data_nodes_.size(); ++i) {
      const auto &data_node = host_queue_ds_actor->data_nodes_[i];
      MS_EXCEPTION_IF_NULL(data_node);
      const auto &device_tensor = AnfAlgo::GetMutableOutputAddr(data_node, 0, false);
      MS_EXCEPTION_IF_NULL(device_tensor);
      ofs << "\t\t\tnode_order_number:" << i << "\tnode_name:" << data_node->fullname_with_scope()
          << "\tptr:" << device_tensor->GetPtr() << "\tsize:" << device_tensor->GetSize()
          << "\toriginal_ref_count:" << device_tensor->original_ref_count() << "\n ";
    }
  }

  ofs << "\t\toutput_data_arrows:" << actor->output_op_arrows_.size() << "\n ";
  for (const auto &data_arrow : actor->output_op_arrows_) {
    MS_EXCEPTION_IF_NULL(data_arrow);
    ofs << "\t\t\tfrom_output_index:" << data_arrow->from_output_index_
        << "\tto_actor_name:" << data_arrow->to_op_id_.Name() << "\tto_input_index:" << data_arrow->to_input_index_
        << "\n";
  }

  ofs << "\t\toutput_result_arrows:" << actor->output_result_arrows_.size() << "\n ";
  for (const auto &result_arrow : actor->output_result_arrows_) {
    MS_EXCEPTION_IF_NULL(result_arrow);
    ofs << "\t\t\tfrom_output_index:" << result_arrow->from_output_index_
        << "\tto_actor_name:" << result_arrow->to_op_id_.Name()
        << "\toutput_node_position:" << result_arrow->to_input_index_ << "\n";
  }
}

void GraphScheduler::DumpLoopCountActor(const LoopCountActor *actor, std::ofstream &ofs) const {
  MS_EXCEPTION_IF_NULL(actor);
  ofs << "\tactor_name:" << actor->GetAID().Name() << "\tloop_count:" << actor->loop_count_
      << "\tinput_controls_num:" << actor->input_controls_num_ << "\n";

  ofs << "\t\toutput_control_arrows:" << (actor->data_source_aids_.size() + actor->no_input_kernel_aids_.size() + 1)
      << "\n ";
  for (const auto &aid : actor->data_source_aids_) {
    ofs << "\t\t\tto_actor_name:" << aid.Name() << "\n";
  }
  for (const auto &aid : actor->no_input_kernel_aids_) {
    ofs << "\t\t\tto_actor_name:" << aid.Name() << "\n";
  }
  ofs << "\t\t\tto_actor_name:" << actor->output_aid_.Name() << "\n";
}

void GraphScheduler::DumpKernelActor(const KernelActor *actor, std::ofstream &ofs) const {
  MS_EXCEPTION_IF_NULL(actor);
  MS_EXCEPTION_IF_NULL(actor->device_context_);
  ofs << "\tactor_name:" << actor->GetAID().Name()
      << "\tdevice_context:" << actor->device_context_->device_context_key().ToString()
      << "\tinput_data_num:" << actor->input_datas_num_ << "\tinput_controls_num:" << actor->input_controls_num_
      << "\n";

  const auto &kernel = actor->kernel_;
  MS_EXCEPTION_IF_NULL(kernel);
  ofs << "\t\tkernel_name:" << kernel->fullname_with_scope() << "\tinput_number:" << AnfAlgo::GetInputTensorNum(kernel)
      << "\toutput_number:" << AnfAlgo::GetOutputTensorNum(kernel) << "\n";
  for (size_t i = 0; i < AnfAlgo::GetOutputTensorNum(kernel); ++i) {
    const auto &device_tensor = AnfAlgo::GetMutableOutputAddr(kernel, i, false);
    MS_EXCEPTION_IF_NULL(device_tensor);
    ofs << "\t\t\toutput_index:" << i << "\tptr:" << device_tensor->GetPtr() << "\tsize:" << device_tensor->GetSize()
        << "\toriginal_ref_count:" << device_tensor->original_ref_count() << "\n ";
  }

  ofs << "\t\tdevice_tensor_stores:" << actor->device_tensor_store_keys_.size() << "\n ";
  for (const auto &device_tensor_store_key : actor->device_tensor_store_keys_) {
    const auto &node = reinterpret_cast<AnfNode *>(device_tensor_store_key.second);
    MS_EXCEPTION_IF_NULL(node);
    ofs << "\t\t\tto_input_index:" << device_tensor_store_key.first
        << "\tfrom_node_name:" << node->fullname_with_scope() << "\n";
  }

  ofs << "\t\toutput_data_arrows:" << actor->output_op_arrows_.size() << "\n ";
  for (const auto &data_arrow : actor->output_op_arrows_) {
    MS_EXCEPTION_IF_NULL(data_arrow);
    ofs << "\t\t\tfrom_output_index:" << data_arrow->from_output_index_
        << "\tto_actor_name:" << data_arrow->to_op_id_.Name() << "\tto_input_index:" << data_arrow->to_input_index_
        << "\n";
  }

  ofs << "\t\toutput_control_arrows:" << actor->output_op_controls_.size() << "\n ";
  for (const auto &aid : actor->output_op_controls_) {
    ofs << "\t\t\tto_actor_name:" << aid.Name() << "\n";
  }

  ofs << "\t\toutput_result_arrows:" << actor->output_result_arrows_.size() << "\n ";
  for (const auto &result_arrow : actor->output_result_arrows_) {
    MS_EXCEPTION_IF_NULL(result_arrow);
    ofs << "\t\t\tfrom_output_index:" << result_arrow->from_output_index_
        << "\tto_actor_name:" << result_arrow->to_op_id_.Name()
        << "\toutput_node_position:" << result_arrow->to_input_index_ << "\n";
  }
}

void GraphScheduler::DumpOutputActor(const OutputActor *actor, std::ofstream &ofs) const {
  MS_EXCEPTION_IF_NULL(actor);
  ofs << "\tactor_name:" << actor->GetAID().Name() << "\tloop_count:" << actor->loop_count_
      << "\toutputs_num:" << actor->outputs_num_ << "\n";

  ofs << "\t\tdevice_tensor_store_keys:" << actor->device_tensor_store_keys_.size() << "\n ";
  for (const auto &device_tensor_store_key : actor->device_tensor_store_keys_) {
    MS_EXCEPTION_IF_NULL(device_tensor_store_key.second);
    ofs << "\t\t\toutput_node_position:" << device_tensor_store_key.first
        << "\toutput_node_name:" << device_tensor_store_key.second->fullname_with_scope() << "\n";
  }
}

}  // namespace runtime
}  // namespace mindspore
