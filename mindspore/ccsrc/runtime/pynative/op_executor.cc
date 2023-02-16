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

#include "runtime/pynative/op_executor.h"

namespace mindspore::runtime {
OpExecutor &OpExecutor::GetInstance() {
  static OpExecutor instance;
  return instance;
}

OpExecutor::OpExecutor() = default;

OpExecutor::~OpExecutor() = default;

void OpExecutor::Register(const std::function<void()> &callback) { batch_build_callback_ = callback; }

void OpExecutor::Reset() {
  ClearResources();
  batch_build_callback_ = nullptr;
  async_queue_.Reset();
}

void OpExecutor::ClearResources() {
  MS_LOG(DEBUG) << "Start clear tasks";
  // Set the build task failed, and no need to run op_run_tasks.
  for (auto &build_task : op_build_tasks_) {
    build_task->SetBuildReady(false);
  }
  op_build_tasks_.clear();
  MS_LOG(DEBUG) << "End clear tasks";
}

void OpExecutor::WaitForBuild() {
  if (!executing_) {
    ExecuteGuard guard;
    if (batch_build_callback_ != nullptr) {
      batch_build_callback_();
    }
  }
}

void OpExecutor::WaitForRun() {
  MS_LOG(DEBUG) << "Start";
  async_queue_.Wait();
  MS_LOG(DEBUG) << "All task finish";
}

void OpExecutor::Wait() {
  if (PyGILState_Check() != 0) {
    py::gil_scoped_release gil;
    WaitForBuild();
    WaitForRun();
  } else {
    WaitForBuild();
    WaitForRun();
  }
}

void OpExecutor::PushOpBuildTask(const std::shared_ptr<pynative::BackendOpBuildTask> &op_build_task) {
  op_build_tasks_.push_back(op_build_task);
}

void OpExecutor::PushOpRunTask(const std::shared_ptr<pynative::BackendOpRunTask> &op_run_task) {
  async_queue_.Push(op_run_task);
  (void)actor_in_queue_.insert(op_run_task->context()->graph_id());
}

void OpExecutor::ClearOpBuildTasks() {
  for (auto &task : op_build_tasks_) {
    task->SetBuildReady(true);
  }
  op_build_tasks_.clear();
  MS_LOG(DEBUG) << "Clear build task";
}

bool OpExecutor::BuildQueueEmpty() { return op_build_tasks_.empty(); }

bool OpExecutor::RunQueueEmpty() { return async_queue_.Empty(); }

bool OpExecutor::BuildQueueFull() { return op_build_tasks_.size() > kMaxQueueSize; }

bool OpExecutor::ActorInQueue(GraphId graph_id) {
  auto iter = actor_in_queue_.find(graph_id);
  return iter != actor_in_queue_.end();
}

void OpExecutor::WorkerJoin() {
  Wait();
  async_queue_.WorkerJoin();
}

bool OpExecutor::BuildInQueue(GraphId graph_id) {
  return std::any_of(op_build_tasks_.begin(), op_build_tasks_.end(),
                     [=](const std::shared_ptr<pynative::BackendOpBuildTask> &backend_op_build_task) {
                       return backend_op_build_task->context()->graph_id() == graph_id;
                     });
}
}  // namespace mindspore::runtime
