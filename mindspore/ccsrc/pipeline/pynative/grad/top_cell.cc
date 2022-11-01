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
#include "pipeline/pynative/grad/top_cell.h"
#include <iostream>
#include "pipeline/pynative/pynative_utils.h"
#include "ir/tensor.h"
#include "runtime/device/device_address.h"

namespace mindspore {
namespace pynative {
namespace {
void SplitString(const std::string &str, std::vector<std::string> *id_vec) {
  constexpr char colon_delim = ':';
  constexpr char angle_bracket_left_delim = '<';
  constexpr char angle_bracket_right_delim = '>';
  auto paren_pos = str.find_first_of(angle_bracket_left_delim);
  if (paren_pos == std::string::npos) {
    MS_LOG(EXCEPTION) << "Get wrong str " << str;
  }
  size_t str_size = str.size();
  const auto &sub_str = str.substr(paren_pos + 1, str_size - paren_pos - 2);
  MS_LOG(DEBUG) << "Ori str " << str << ", get sub str " << sub_str;
  auto begin = sub_str.find_first_not_of(colon_delim, 0);
  auto end = sub_str.find_first_of(colon_delim, begin);
  MS_EXCEPTION_IF_NULL(id_vec);
  while (end != std::string::npos || begin != std::string::npos) {
    (void)id_vec->emplace_back(sub_str.substr(begin, end - begin));
    begin = sub_str.find_first_not_of(colon_delim, end);
    end = sub_str.find_first_of(colon_delim, begin);
    paren_pos = sub_str.find_first_of(angle_bracket_left_delim, begin);
    if (paren_pos < end) {
      end = sub_str.find_last_of(angle_bracket_right_delim) + 1;
    }
  }
}
}  // namespace

void TopCellInfo::RecordCellBackwardHookOp(const std::string &cell_order, const AnfNodePtr &hook_op) {
  MS_EXCEPTION_IF_NULL(hook_op);
  (void)cell_backward_hook_op_[cell_order].emplace_back(hook_op);
  constexpr size_t cell_backward_hook_max_num = 2;
  if (cell_backward_hook_op_[cell_order].size() > cell_backward_hook_max_num) {
    MS_LOG(EXCEPTION) << "Cell order: " << cell_order << " only has two backward hook op.";
  }
}

void TopCellInfo::CheckSubCellHookChanged() { sub_cell_hook_changed_.clear(); }

void TopCellInfo::ClearDeviceMemory() const {
  MS_LOG(DEBUG) << "Clear device memory in value nodes of bprop graph, top cell: " << cell_id_;
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  const auto &device_target = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  if (device_target == kCPUDevice) {
    MS_LOG(DEBUG) << "No need to clear device address when run in CPU device.";
    return;
  }
  // Get all tensors obj in value node of running graph
  std::vector<tensor::TensorPtr> tensors_in_bprop_graph;
  MS_EXCEPTION_IF_NULL(resource_);
  const auto &bprop_graph = resource_->func_graph();
  MS_EXCEPTION_IF_NULL(bprop_graph);
  const auto &value_node_list = bprop_graph->value_nodes();
  for (const auto &elem : value_node_list) {
    auto &node = elem.first;
    MS_EXCEPTION_IF_NULL(node);
    auto value_node = node->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(value_node);
    TensorValueToTensor(value_node->value(), &tensors_in_bprop_graph);
  }
  for (const auto &tensor : tensors_in_bprop_graph) {
    MS_EXCEPTION_IF_NULL(tensor);
    MS_LOG(DEBUG) << "Clear device address for tensor: " << tensor->ToString();
    auto device_sync = tensor->device_address();
    auto device_address = std::dynamic_pointer_cast<device::DeviceAddress>(device_sync);
    if (device_address == nullptr) {
      continue;
    }
    if (!device_address->from_persistent_mem()) {
      tensor->set_device_address(nullptr);
    }
  }
}

void TopCellInfo::DeleteParamNodeInfo(const FuncGraphPtr &g, const std::string &id) {
  auto &graph_info = graph_info_map().at(g);
  MS_EXCEPTION_IF_NULL(graph_info);
  graph_info->input_params.erase(id);
}

void TopCellInfo::SetParamNodeMapInGraphInfoMap(const FuncGraphPtr &g, const std::string &id, const ParameterPtr &param,
                                                bool is_weight) const {
  auto &graph_info = graph_info_map().at(g);
  MS_EXCEPTION_IF_NULL(graph_info);
  if (is_weight) {
    graph_info->weight_params[id] = param;
  } else {
    graph_info->input_params[id] = param;
  }
}

void TopCellInfo::SetNodeMapInGraphInfoMap(const FuncGraphPtr &g, const std::string &id, const AnfNodePtr &node,
                                           int64_t index) const {
  auto &graph_info = graph_info_map().at(g);
  MS_EXCEPTION_IF_NULL(graph_info);
  graph_info->node_map[id] = std::make_pair(node, std::vector<int64_t>{index});
  // For example, set id of ((A,B),C) = {CNode, -1}
  SetMultipleOutputToGraphInfoMap(g, id, node);
}

void TopCellInfo::SetMultipleOutputToGraphInfoMap(const FuncGraphPtr &g, const string &id,
                                                  const AnfNodePtr &node) const {
  if (id.find("Tuple") == std::string::npos && id.find("List") == std::string::npos) {
    return;
  }
  std::vector<std::string> id_vec;
  SplitString(id, &id_vec);
  auto tuple_size = static_cast<int64_t>(id_vec.size());
  for (int64_t i = 0; i < tuple_size; ++i) {
    // Set id of (A,B) = {CNode, 0}; Set id of C = {CNode, 1}
    SetNodeMapInGraphInfoMap(g, id_vec[i], node, i);
    SetNestedMultipleOutputToGraphInfoMap(g, id_vec[i], node, std::vector<int64_t>{i});
  }
}

void TopCellInfo::SetNestedMultipleOutputToGraphInfoMap(const FuncGraphPtr &g, const string &id, const AnfNodePtr &node,
                                                        const std::vector<int64_t> &index_sequence) const {
  if (id.find("Tuple") == std::string::npos && id.find("List") == std::string::npos) {
    return;
  }
  MS_EXCEPTION_IF_NULL(node);
  std::vector<std::string> id_vec;
  SplitString(id, &id_vec);
  auto tuple_size = static_cast<int64_t>(id_vec.size());
  for (int64_t i = 0; i < tuple_size; ++i) {
    std::vector<int64_t> tmp = index_sequence;
    (void)tmp.emplace_back(i);
    // Set id of A = {CNode, [0, 0]}; Set id of B = {CNode, [0, 1]};
    SetUnpackOutputToGraphInfoMap(g, id_vec[i], node, tmp);
    // If output have more nested tuple or list
    SetNestedMultipleOutputToGraphInfoMap(g, id_vec[i], node, tmp);
  }
}

void TopCellInfo::SetUnpackOutputToGraphInfoMap(const FuncGraphPtr &g, const std::string &id, const AnfNodePtr &node,
                                                const std::vector<int64_t> &index) const {
  auto &graph_info = graph_info_map().at(g);
  MS_EXCEPTION_IF_NULL(graph_info);
  graph_info->node_map[id] = std::make_pair(node, index);
}
}  // namespace pynative
}  // namespace mindspore
