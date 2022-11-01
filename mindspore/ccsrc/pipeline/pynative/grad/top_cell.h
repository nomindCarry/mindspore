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

#ifndef MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_TOP_CELL_H_
#define MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_TOP_CELL_H_

#include <utility>
#include <vector>
#include <string>
#include <memory>
#include <mutex>
#include <stack>
#include <set>
#include <map>
#include "include/common/utils/convert_utils.h"
#include "utils/hash_map.h"
#include "utils/hash_set.h"
#include "pybind11/numpy.h"
#include "pybind11/pytypes.h"
#include "pybind_api/ir/base_ref_py.h"
#include "ir/anf.h"
#include "frontend/optimizer/ad/kpynative.h"
#include "frontend/operator/composite/composite.h"
#include "pipeline/jit/resource.h"
#include "pipeline/pynative/base.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace pynative {
namespace py = pybind11;
class GradExecutor;
using CellIdWithBackwardHookOp = mindspore::HashMap<std::string, std::vector<AnfNodePtr>>;

struct GraphInfo {
  OrderedMap<std::string, ParameterPtr> input_params;   // Hold input parameters
  OrderedMap<std::string, ParameterPtr> weight_params;  // Hold weights parameters
  // Hold op op output or combination of output
  mindspore::HashMap<std::string, std::pair<AnfNodePtr, std::vector<int64_t>>> node_map;
};
using GraphInfoPtr = std::shared_ptr<GraphInfo>;

class TopCellInfo {
 public:
  ~TopCellInfo() = default;
  TopCellInfo(size_t grad_order, std::string cellid, std::string already_run_cell_id, pipeline::ResourcePtr r,
              FuncGraphPtr fg)
      : grad_order_(grad_order),
        cell_id_(std::move(cellid)),
        already_run_cell_id_(std::move(already_run_cell_id)),
        resource_(std::move(r)),
        fg_(std::move(fg)) {}

  bool is_init_kpynative() const { return is_init_kpynative_; }
  void set_init_kpynative(bool init) { is_init_kpynative_ = init; }
  size_t grad_order() const { return grad_order_; }
  bool hook_changed() const { return hook_changed_; }
  void set_hook_changed(bool hook_changed) { hook_changed_ = hook_changed; }
  void set_sub_cell_hook_changed(const std::string &sub_cell) { (void)sub_cell_hook_changed_.emplace(sub_cell); }
  const CellIdWithBackwardHookOp &cell_backward_hook_op() const { return cell_backward_hook_op_; }
  void RecordCellBackwardHookOp(const std::string &cell_order, const AnfNodePtr &hook_op);
  void ClearCellHookOp() { cell_backward_hook_op_.clear(); }
  bool ms_function_flag() const { return ms_function_flag_; }
  void set_ms_function_flag(bool ms_function_flag) { ms_function_flag_ = ms_function_flag; }
  bool forward_already_run() const { return forward_already_run_; }
  void set_forward_already_run(bool set_forward_already_run) { forward_already_run_ = set_forward_already_run; }
  pipeline::ResourcePtr resource() const { return resource_; }
  inline FuncGraphPtr fg() const {
    MS_EXCEPTION_IF_NULL(fg_);
    return fg_;
  }
  void set_fg(const FuncGraphPtr &fg) { fg_ = fg; }
  const std::string &cell_id() const { return cell_id_; }
  const std::string &already_run_cell_id() const { return already_run_cell_id_; }
  void set_input_args_id(const std::string &input_args_id) { input_args_id_ = input_args_id; }
  const std::string &input_args_id() const { return input_args_id_; }
  const std::string &grad_operation() const { return grad_operation_; }
  void set_grad_operation(const std::string &grad_operation) { grad_operation_ = grad_operation; }
  void set_input_args_info(const InputArgsInfoPtr &input_args_info) { input_args_info_ = input_args_info; }
  void CheckSubCellHookChanged();
  void SetGraphInfoMap(const FuncGraphPtr &fg, const GraphInfoPtr &graph_info) { graph_info_map_[fg] = graph_info; }
  const OrderedMap<FuncGraphPtr, GraphInfoPtr> &graph_info_map() const { return graph_info_map_; }
  inline ad::KPynativeCellPtr k_pynative_cell_ptr() const {
    MS_EXCEPTION_IF_NULL(k_pynative_cell_ptr_);
    return k_pynative_cell_ptr_;
  }
  void set_k_pynative_cell_ptr(const ad::KPynativeCellPtr &k_pynative_cell_ptr) {
    k_pynative_cell_ptr_ = k_pynative_cell_ptr;
  }
  void DeleteParamNodeInfo(const FuncGraphPtr &g, const std::string &id);
  void SetParamNodeMapInGraphInfoMap(const FuncGraphPtr &g, const std::string &id, const ParameterPtr &param,
                                     bool is_weight = false) const;
  void SetNodeMapInGraphInfoMap(const FuncGraphPtr &g, const std::string &id, const AnfNodePtr &node,
                                int64_t index = -1) const;
  void ClearDeviceMemory() const;

 private:
  void SetMultipleOutputToGraphInfoMap(const FuncGraphPtr &g, const string &id, const AnfNodePtr &node) const;
  void SetNestedMultipleOutputToGraphInfoMap(const FuncGraphPtr &g, const string &id, const AnfNodePtr &node,
                                             const std::vector<int64_t> &index_sequence) const;
  void SetUnpackOutputToGraphInfoMap(const FuncGraphPtr &g, const std::string &id, const AnfNodePtr &node,
                                     const std::vector<int64_t> &index) const;

  bool hook_changed_{false};
  bool ms_function_flag_{false};
  bool is_init_kpynative_{false};
  bool forward_already_run_{false};
  size_t grad_order_{0};
  std::string cell_id_;
  std::string already_run_cell_id_;
  std::string input_args_id_;
  std::string grad_operation_;
  InputArgsInfoPtr input_args_info_{nullptr};
  pipeline::ResourcePtr resource_{nullptr};
  FuncGraphPtr fg_{nullptr};
  ad::KPynativeCellPtr k_pynative_cell_ptr_{nullptr};
  OrderedMap<FuncGraphPtr, GraphInfoPtr> graph_info_map_;
  // Record `register hook` or `remove hook` function has been called by sub cell
  // The record range between the begin and end of top cell.
  mindspore::HashSet<std::string> sub_cell_hook_changed_;
  // Record backward hook ops for each cell object.
  // Each cell object has two backward hook ops.
  CellIdWithBackwardHookOp cell_backward_hook_op_;
};
using TopCellInfoPtr = std::shared_ptr<TopCellInfo>;
}  // namespace pynative
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_TOP_CELL_H_
