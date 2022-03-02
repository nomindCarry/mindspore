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

#ifndef MINDSPORE_CCSRC_RUNTIME_GRAPH_SCHEDULER_RPC_NODE_SCHEDULER_H_
#define MINDSPORE_CCSRC_RUNTIME_GRAPH_SCHEDULER_RPC_NODE_SCHEDULER_H_

#include <vector>
#include <string>
#include <memory>
#include "runtime/graph_scheduler/actor/actor_set.h"
#include "runtime/graph_scheduler/graph_compiler.h"

namespace mindspore {
namespace runtime {
using mindspore::device::DeviceContext;
using mindspore::session::KernelGraph;
using mindspore::session::KernelWithIndex;

// Scheduler for rpc actors, e.g., it adds inter-process arrows, generate router for actors, etc.
class RpcNodeScheduler {
 public:
  RpcNodeScheduler() : rpc_actor_set_(nullptr) {}
  ~RpcNodeScheduler() = default;

  // Create rpc actor set.
  void Initialize();

  // Build rpc actors and return rpc actor set.
  RpcActorSetPtr Build(const GraphCompilerInfo &graph_compiler_info);

  // Link rpc actors with inter-process arrows.
  void Link(const ActorSet *actor_set);

  // This should be called by 'GraphScheduler::Scheduler()' method.
  // Used to start servers for recv actors and create connections for send actors.
  void Schedule();

  // Insert Send/Recv actors generated by GraphScheduler to the rpc actor set.
  void InsertSendActor(const SendActorPtr &send_actor);
  void InsertRecvActor(const RecvActorPtr &recv_actor);

  // Set op context to rpc actors.
  void SetOpcontext(OpContext<DeviceTensor> *const op_context);

  // Reset op context for rpc actors.
  void ResetOpcontext();

 private:
  // Create new route table proxy.
  ActorRouteTableProxyPtr CreateRouteTableProxy();

  RpcActorSetPtr rpc_actor_set_;
};

// The setter of op context for rpc actors.
class RpcActorOpContextSetter {
 public:
  explicit RpcActorOpContextSetter(RpcNodeScheduler *rpc_node_scheduler, OpContext<DeviceTensor> *const op_context)
      : rpc_node_scheduler_(rpc_node_scheduler), op_context_(op_context) {
    rpc_node_scheduler_->SetOpcontext(op_context_);
  }
  ~RpcActorOpContextSetter() { rpc_node_scheduler_->ResetOpcontext(); }

 private:
  RpcNodeScheduler *rpc_node_scheduler_;
  OpContext<DeviceTensor> *op_context_;
};
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_GRAPH_SCHEDULER_RPC_NODE_SCHEDULER_H_
