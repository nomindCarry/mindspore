/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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

#include "extendrt/delegate/ascend_ge/ge_graph_executor.h"
#include <tuple>
#include <algorithm>
#include <utility>
#include "extendrt/delegate/factory.h"
#include "include/common/utils/scoped_long_running.h"
#include "include/api/context.h"
#include "include/api/status.h"
#include "include/transform/graph_ir/utils.h"
#include "include/backend/device_type.h"
#include "runtime/device/ms_device_shape_transfer.h"
#include "src/common/common.h"
#include "src/common/file_utils.h"

namespace mindspore {
namespace {
constexpr auto kProviderGe = "ge";
constexpr auto kDump = "dump";
constexpr auto kDumpMode = "dump_mode";
constexpr auto kProfiling = "profiler";

std::string GetGraphName(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  KernelGraphPtr kg = std::dynamic_pointer_cast<session::KernelGraph>(graph);
  std::string name;
  if (kg == nullptr) {
    name = graph->ToString();
  } else {
    FuncGraphPtr origin_graph = kg->GetFuncGraph();
    MS_EXCEPTION_IF_NULL(origin_graph);
    name = origin_graph->ToString();
  }
  return name;
}

void GetMeRetDataType(const AbstractBasePtr &cnode_data, std::vector<TypeId> *me_types) {
  MS_EXCEPTION_IF_NULL(cnode_data);

  if (cnode_data->isa<abstract::AbstractTensor>()) {
    TypeId me_type = cnode_data->BuildType()->type_id();
    if (me_type == kObjectTypeTensorType) {
      me_type = dyn_cast<TensorType>(cnode_data->BuildType())->element()->type_id();
      me_types->emplace_back(me_type);
    }
    return;
  }
  if (cnode_data->isa<abstract::AbstractScalar>()) {
    TypeId me_type = cnode_data->BuildType()->type_id();
    me_types->emplace_back(me_type);
  }
  auto abstract_tuple = cnode_data->cast<abstract::AbstractTuplePtr>();
  MS_EXCEPTION_IF_NULL(abstract_tuple);
  auto elements = abstract_tuple->elements();
  for (size_t i = 0; i < abstract_tuple->size(); ++i) {
    GetMeRetDataType(elements[i], me_types);
  }
}

transform::TensorOrderMap GetParams(const FuncGraphPtr &anf_graph) {
  MS_EXCEPTION_IF_NULL(anf_graph);
  transform::TensorOrderMap res;
  for (auto &anf_node : anf_graph->parameters()) {
    MS_EXCEPTION_IF_NULL(anf_node);
    auto para = anf_node->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(para);
    if (para->has_default()) {
      auto value = para->default_param();
      MS_EXCEPTION_IF_NULL(value);
      auto tensor = value->cast<std::shared_ptr<tensor::Tensor>>();
      res.emplace(para->name(), tensor);
      MS_LOG(INFO) << "Parameter " << para->name() << " has default value.";
    }
  }
  return res;
}

bool AddDFGraph(const FuncGraphPtr &anf_graph, const transform::TensorOrderMap &init_inputs_map,
                const std::map<std::string, std::string> &ge_options, bool export_air) {
  MS_EXCEPTION_IF_NULL(anf_graph);
  auto converter = transform::NewConverter(anf_graph);
  if (export_air) {
    MS_LOG(INFO) << "Set DfGraphConvertor training : false";
    transform::SetTraining(converter, false);
  }
  transform::BuildGraph(anf_graph->ToString(), converter, init_inputs_map);
  transform::GenerateBroadcastGraph(converter, init_inputs_map);
  transform::GenerateCheckpointGraph(converter);
  auto err_code = transform::ErrCode(converter);
  if (err_code != 0) {
    transform::ClearGraph();
    MS_LOG(ERROR) << "Convert df graph failed, err:" << err_code;
    return false;
  }

  std::string graph_name = anf_graph->ToString();
  std::string init_graph = "init_subgraph." + graph_name;
  std::string checkpoint_name = "save." + GetGraphName(anf_graph);
  auto options = ge_options;
  if (common::GetEnv("GE_TRAIN") == "1") {
    options["ge.exec.variable_acc"] = "1";
  }
  (void)transform::AddGraph(graph_name, transform::GetComputeGraph(converter), options);
  (void)transform::AddGraph(init_graph, transform::GetInitGraph(converter));
  (void)transform::AddGraph(BROADCAST_GRAPH_NAME, transform::GetBroadcastGraph(converter));

  transform::Status ret = transform::AddGraph(checkpoint_name, transform::GetSaveCheckpointGraph(converter));
  if (ret == transform::Status::SUCCESS) {
    transform::SetAnfGraph(checkpoint_name, anf_graph);
  }
  return true;
}

void CreateSessionAndGraphRunner(const std::map<std::string, std::string> &ge_options) {
  std::shared_ptr<::ge::Session> sess = transform::GetGeSession();
  if (sess == nullptr) {
    transform::SessionOptions options = ge_options;
    options["ge.trainFlag"] = "0";
    options["ge.enablePrintOpPass"] = "0";
    sess = transform::NewSession(options);
    transform::SetGeSession(sess);
  }

  transform::GraphRunnerOptions options;
  options.sess_ptr = sess;
  auto graph_runner = transform::NewGraphRunner(options);
  if (graph_runner == nullptr) {
    MS_LOG(ERROR) << "Create new graph runner failed";
  } else {
    transform::SetGraphRunner(graph_runner);
  }
}

void RunGeInitGraph(const FuncGraphPtr &anf_graph) {
  MS_LOG(DEBUG) << "ExecInitGraph start.";

  std::vector<transform::GeTensorPtr> ge_outputs;
  transform::RunOptions run_options;

  run_options.name = "init_subgraph." + anf_graph->ToString();
  if (transform::GetGraphByName(run_options.name) == nullptr) {
    MS_LOG(WARNING) << "Can not find " << run_options.name
                    << " sub graph, don't need data init subgraph in INFER mode.";
    return;
  }
  auto graph_runner = transform::GetGraphRunner();
  if (graph_runner == nullptr) {
    MS_LOG(EXCEPTION) << "Can not found GraphRunner.";
  }

  std::vector<transform::GeTensorPtr> ge_tensors;
  {
    // Release GIL before calling into (potentially long-running) C++ code
    mindspore::ScopedLongRunning long_running;
    transform::Status ret = transform::RunGraph(graph_runner, run_options, ge_tensors, &ge_outputs);
    if (ret != transform::Status::SUCCESS) {
      MS_LOG(EXCEPTION) << "Exec " << run_options.name << " graph failed.";
    }
    MS_LOG(INFO) << "Exec " << run_options.name << " graph success.";
  }
}

void GetGeSessionOptions(const ConfigInfos &config_infos, std::map<std::string, std::string> *ge_options) {
  MS_EXCEPTION_IF_NULL(ge_options);
  if (config_infos.find(lite::kAscendContextSection) == config_infos.end()) {
    return;
  }
  auto config = config_infos.at(lite::kAscendContextSection);
  if (config.find(lite::kDumpPathKey) != config.end()) {
    auto dump_path = config.at(lite::kDumpPathKey);
    auto real_path = lite::RealPath(dump_path.c_str());
    std::ifstream ifs(real_path);
    if (!ifs.good() || !ifs.is_open()) {
      MS_LOG(EXCEPTION) << "The dump config file: " << real_path << " is not exit or open failed.";
    }
    nlohmann::json dump_cfg_json;
    try {
      dump_cfg_json = nlohmann::json::parse(ifs);
    } catch (const nlohmann::json::parse_error &error) {
      MS_LOG(EXCEPTION) << "parse json failed, please check the file: " << real_path;
    }
    if (dump_cfg_json[kDump] != nullptr && dump_cfg_json[kDump][kDumpMode] != nullptr) {
      (*ge_options)["ge.exec.enableDump"] = "1";
      (*ge_options)["ge.exec.dumpMode"] = dump_cfg_json[kDump][kDumpMode].get<std::string>();
    }
  }
  if (config.find(lite::kProfilingPathKey) != config.end()) {
    auto profiling_path = config.at(lite::kProfilingPathKey);
    auto real_path = lite::RealPath(profiling_path.c_str());
    std::ifstream ifs(real_path);
    if (!ifs.good() || !ifs.is_open()) {
      MS_LOG(EXCEPTION) << "The profiling_path config file: " << real_path << " is not exit or open failed.";
    }
    nlohmann::json profiling_cfg_json;
    try {
      profiling_cfg_json = nlohmann::json::parse(ifs);
    } catch (const nlohmann::json::parse_error &error) {
      MS_LOG(EXCEPTION) << "parse json failed, please check the file: " << real_path;
    }
    if (profiling_cfg_json[kProfiling] != nullptr) {
      (*ge_options)["ge.exec.profilingMode"] = "1";
      (*ge_options)["ge.exec.profilingOptions"] = profiling_cfg_json[kProfiling].dump();
    }
  }

  if (config_infos.find(lite::kAscendContextSection) != config_infos.end()) {
    auto ascend_context = config_infos.at(lite::kAscendContextSection);
    if (ascend_context.find(lite::kGeVariableMemoryMaxSize) != ascend_context.end()) {
      auto variable_memory_max_size = ascend_context[lite::kGeVariableMemoryMaxSize];
      (*ge_options)["ge.variableMemoryMaxSize"] = variable_memory_max_size;
    }
    if (ascend_context.find(lite::kGeGraphMemoryMaxSize) != ascend_context.end()) {
      auto graph_memory_max_size = ascend_context[lite::kGeGraphMemoryMaxSize];
      (*ge_options)["ge.graphMemoryMaxSize"] = graph_memory_max_size;
    }
  }

  (*ge_options)["ge.graph_compiler_cache_dir"] =
    config.find(lite::kGraphCompilerCacheDirKey) == config.end() ? "" : config.at(lite::kGraphCompilerCacheDirKey);
}

void GetGeGraphOptions(const FuncGraphPtr &anf_graph, const std::shared_ptr<Context> &ctx,
                       const ConfigInfos &config_infos, std::map<std::string, std::string> *ge_options) {
  MS_EXCEPTION_IF_NULL(anf_graph);
  MS_EXCEPTION_IF_NULL(ge_options);
  (*ge_options)["ge.graph_key"] = anf_graph->ToString();
  auto device_list = ctx->MutableDeviceInfo();
  auto itr =
    std::find_if(device_list.begin(), device_list.end(), [](const std::shared_ptr<DeviceInfoContext> &device_info) {
      return device_info->GetDeviceType() == DeviceType::kAscend;
    });
  if (itr == device_list.end()) {
    MS_LOG(EXCEPTION) << "Can not find ascend device context.";
  }
  auto ascend_device_info = (*itr)->Cast<AscendDeviceInfo>();
  auto precision_mode = ascend_device_info->GetPrecisionMode();
  if (!precision_mode.empty()) {
    (*ge_options)["ge.exec.precision_mode"] = precision_mode;
  }
  if (config_infos.find(lite::kAscendContextSection) == config_infos.end()) {
    return;
  }
  auto config = config_infos.at(lite::kAscendContextSection);
  (*ge_options)["ge.exec.modify_mixlist"] =
    config.find(lite::kModifyMixList) == config.end() ? "" : config.at(lite::kModifyMixList);
}
}  // namespace

FuncGraphPtr GeGraphExecutor::BuildDFGraph(const FuncGraphPtr &anf_graph, const std::shared_ptr<Context> &ctx,
                                           const transform::TensorOrderMap &init_inputs_map, bool export_air,
                                           const ConfigInfos config_infos) {
  MS_EXCEPTION_IF_NULL(anf_graph);
  std::map<std::string, std::string> graph_options;
  GetGeGraphOptions(anf_graph, ctx, config_infos, &graph_options);
  if (!AddDFGraph(anf_graph, init_inputs_map, graph_options, export_air)) {
    MS_LOG(ERROR) << "GenConvertor failed";
    return nullptr;
  }
  (void)setenv("GE_TRAIN", "0", 1);
  std::map<std::string, std::string> session_options;
  GetGeSessionOptions(config_infos, &session_options);
  CreateSessionAndGraphRunner(session_options);

  auto graph_runner = transform::GetGraphRunner();
  if (graph_runner == nullptr) {
    MS_LOG(ERROR) << "Can not found GraphRunner";
    return nullptr;
  }
  return anf_graph;
}

bool GeGraphExecutor::CompileGraph(const FuncGraphPtr &graph, const std::map<string, string> &compile_options) {
  if (graph == nullptr) {
    MS_LOG(ERROR) << "Input param graph is nullptr.";
    return false;
  }
  KernelGraphPtr kg = std::dynamic_pointer_cast<session::KernelGraph>(graph);
  if (kg == nullptr) {
    MS_LOG(ERROR) << "Dynamic cast kernel graph failed.";
    return false;
  }
  // opt::GeOptimization(origin_graph);
  (void)BuildDFGraph(kg, context_, GetParams(kg), false, config_infos_);
  kg->set_run_mode(device::RunMode::kGraphMode);
  // copy init weight to device
  RunGeInitGraph(kg);
  return true;
}

bool GeGraphExecutor::RunGraph(const FuncGraphPtr &graph, const std::vector<tensor::Tensor> &inputs,
                               std::vector<tensor::Tensor> *outputs,
                               const std::map<string, string> & /* compile_options */) {
  if (graph == nullptr || outputs == nullptr) {
    MS_LOG(ERROR) << " Input param is nullptr.";
    return false;
  }
  auto graph_name = graph->ToString();
  MS_LOG(INFO) << "GE run graph " << graph_name << " start.";
  std::vector<tensor::TensorPtr> input_tensors;
  for (const auto &input : inputs) {
    auto tensor = std::make_shared<tensor::Tensor>(input);
    input_tensors.emplace_back(std::move(tensor));
  }
  auto ge_inputs = transform::ConvertInputTensors(input_tensors, kOpFormat_NCHW);

  // call ge rungraph
  transform::RunOptions run_options;
  run_options.name = graph_name;
  auto graph_runner = transform::GetGraphRunner();
  if (graph_runner == nullptr) {
    MS_LOG(EXCEPTION) << "Can not found GraphRunner.";
  }
  std::vector<transform::GeTensorPtr> ge_outputs;
  {
    // Release GIL before calling into (potentially long-running) C++ code
    mindspore::ScopedLongRunning long_running;
    MS_LOG(DEBUG) << "Run graph begin, inputs size is: " << inputs.size();
    transform::Status ret = transform::RunGraphAsync(graph_runner, run_options, ge_inputs, &ge_outputs);
    MS_LOG(DEBUG) << "Run graph finish, outputs size is: " << ge_outputs.size();
    if (ret != transform::Status::SUCCESS) {
      MS_LOG(EXCEPTION) << "Exec graph failed";
    }
  }

  AnfNodePtr output = graph->get_return()->input(1);
  MS_EXCEPTION_IF_NULL(output);
  std::vector<TypeId> me_types;
  auto output_c = output->cast<CNodePtr>()->abstract();
  // get output node data types
  GetMeRetDataType(output_c, &me_types);
  if (!outputs->empty() && (outputs->size() != ge_outputs.size())) {
    MS_LOG(EXCEPTION) << "Invalid output size, outputs's size " << outputs->size() << "ge tensor size "
                      << ge_outputs.size();
  }
  if (!outputs->empty()) {
    for (size_t i = 0; i < outputs->size(); ++i) {
      const auto &tensor = ge_outputs[i];
      if ((*outputs)[i].Size() < LongToSize(UlongToLong(tensor->GetSize()))) {
        MS_LOG(EXCEPTION) << "Output node " << i << "'s mem size " << (*outputs)[i].DataSize()
                          << " is less than actual output size " << tensor->GetSize();
      }
      if ((*outputs)[i].data_c() == nullptr) {
        MS_LOG(ERROR) << "Output data ptr is nullptr.";
        return false;
      }
      // memcpy_s does not support data that more than 2GB
      (void)memcpy(reinterpret_cast<uint8_t *>((*outputs)[i].data_c()), tensor->GetData(), tensor->GetSize());
    }
  } else {
    MS_LOG(INFO) << "Output is empty.";
    if (me_types.size() != ge_outputs.size()) {
      MS_LOG(EXCEPTION) << "Invalid output size, me_type's size " << me_types.size() << " tensor shape size "
                        << ge_outputs.size();
    }
    for (size_t i = 0; i < me_types.size(); ++i) {
      const auto &tensor = ge_outputs[i];
      auto actual_shapes = tensor->GetTensorDesc().GetShape().GetDims();
      tensor::Tensor output_tensor(me_types[i], actual_shapes);
      if (output_tensor.Size() < LongToSize(UlongToLong(tensor->GetSize()))) {
        MS_LOG(EXCEPTION) << "Output node " << i << "'s mem size " << output_tensor.Size()
                          << " is less than actual output size " << tensor->GetSize();
      }
      (void)memcpy(reinterpret_cast<uint8_t *>(output_tensor.data_c()), tensor->GetData(), tensor->GetSize());
      outputs->push_back(output_tensor);
    }
  }
  MS_LOG(INFO) << "GE run graph end.";
  return true;
}

std::vector<tensor::Tensor> GeGraphExecutor::GetInputInfos(const FuncGraphPtr &) {
  return std::vector<tensor::Tensor>();
}

std::vector<tensor::Tensor> GeGraphExecutor::GetOutputInfos(const FuncGraphPtr &) {
  return std::vector<tensor::Tensor>();
}

static std::shared_ptr<device::GraphExecutor> GeGraphExecutorCreator(const std::shared_ptr<Context> &ctx,
                                                                     const ConfigInfos &config_infos) {
  return std::make_shared<GeGraphExecutor>(ctx, config_infos);
}

REG_DELEGATE(kAscend, kProviderGe, GeGraphExecutorCreator)
}  // namespace mindspore
