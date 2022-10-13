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

#ifndef MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_GRAD_H_
#define MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_GRAD_H_

#include <memory>
#include <string>
#include <utility>
#include <stack>
#include <vector>
#include "pipeline/pynative/base.h"
#include "pipeline/pynative/grad/top_cell.h"
#include "pipeline/pynative/grad/ms_function_grad.h"

namespace mindspore {
namespace pynative {
namespace py = pybind11;
class ForwardExecutor;
using ForwardExecutorPtr = std::shared_ptr<ForwardExecutor>;
using ForwardExecutorWeakPtr = std::weak_ptr<ForwardExecutor>;

class GradExecutor {
 public:
  GradExecutor() = default;
  ~GradExecutor() = default;
  explicit GradExecutor(const ForwardExecutorPtr &forward_executor = nullptr)
      : forward_executor_(ForwardExecutorWeakPtr(forward_executor)), ms_function_(std::make_shared<MsFunction>()) {}

  std::function<void(const py::object &, const py::args &)> InitGraph = [this](auto &&PH1, auto &&PH2) {
    NewGraphInner(std::forward<decltype(PH1)>(PH1), std::forward<decltype(PH2)>(PH2));
  };
  std::function<void(const py::object &, const py::object &, const py::args &)> LinkGraph = [this](auto &&PH1,
                                                                                                   auto &&PH2,
                                                                                                   auto &&PH3) {
    EndGraphInner(std::forward<decltype(PH1)>(PH1), std::forward<decltype(PH2)>(PH2), std::forward<decltype(PH3)>(PH3));
  };
  std::function<void(const prim::GradOperationPtr &, const py::object &, const py::object &, const py::object &,
                     const py::args &)>
    GradGraph = [this](auto &&PH1, auto &&PH2, auto &&PH3, auto &&PH4, auto &&PH5) {
      GradNetInner(std::forward<decltype(PH1)>(PH1), std::forward<decltype(PH2)>(PH2), std::forward<decltype(PH3)>(PH3),
                   std::forward<decltype(PH4)>(PH4), std::forward<decltype(PH5)>(PH5));
    };
  std::function<py::object(void)> RunGraph = [this]() { return RunGradGraph(); };
  inline TopCellInfoPtr top_cell() const {
    MS_EXCEPTION_IF_NULL(top_cell_);
    return top_cell_;
  }
  inline MsFunctionPtr ms_function() const {
    MS_EXCEPTION_IF_NULL(ms_function_);
    return ms_function_;
  }
  bool need_renormalize() const { return need_renormalize_; }
  void set_top_cell(TopCellInfoPtr top_cell) { top_cell_ = std::move(top_cell); }
  bool grad_flag() const { return grad_flag_; }
  void set_grad_flag(bool flag) { grad_flag_ = flag; }
  // Construct grad graph for ms_function
  bool eliminate_forward() const { return eliminate_forward_; }
  void set_eliminate_forward(bool eliminate_forward) { eliminate_forward_ = eliminate_forward; }
  void SetHookChanged(const py::object &cell) const;
  void SetGradOrder(const py::object &obj, const py::args &args, const std::string &cell_id);
  void GradNetInner(const prim::GradOperationPtr &grad, const py::object &obj, const py::object &weights,
                    const py::object &grad_position, const py::args &args);
  py::object RunGradGraph();
  CNodePtr ConstructForwardGraph(const FrontendOpRunInfoPtr &op_run_info) const;
  py::object CheckAlreadyRun(const prim::GradOperationPtr &grad, const py::object &obj, const py::object &grad_hash_id,
                             const py::args &args);
  std::string GetAlreadyRunCellId(const std::string &cell_id) const;
  void ProcessOpGradInfo(const FrontendOpRunInfoPtr &op_run_info) const;
  void EndGraphInner(const py::object &obj, const py::object &out, const py::args &args);
  void EndGraphImpl(const InputArgsInfoPtr &input_args_info);
  AnfNodePtr GetInput(const ValuePtr &v) const;
  AnfNodePtr GetParamInput(const ValuePtr &v, const std::string &id) const;
  void ClearRes();

 private:
  ForwardExecutorPtr forward() const;
  FuncGraphPtr curr_g() const { return top_cell()->fg(); }
  void PushHighOrderGraphStack(const TopCellInfoPtr &top_cell);
  TopCellInfoPtr GetTopCell(const string &already_run_cell_id);
  std::string GetCurCellOrder() const;
  void SaveOutputNodeMap(const std::string &obj_id, const FrontendOpRunInfoPtr &op_run_info,
                         const CNodePtr &cnode) const;
  void DoOpGrad(const FrontendOpRunInfoPtr &op_run_info, const CNodePtr &cnode, const ValuePtr &op_out) const;
  AnfNodePtr GetRealInputNodeBySkipHook(const AnfNodePtr &input_node) const;
  void EraseTopCellFromTopCellList(const TopCellInfoPtr &top_cell);
  void SetBpropGraphJitLevel(const py::object &obj) const;
  void ClearGlobalRes();
  void ClearTopCellRes();

  // Higher derivative
  inline bool IsNestedGrad() const;
  void SwitchTopCell();
  void DoParameterReplace(const FuncGraphPtr &first_grad_fg, const std::vector<ValuePtr> &forward_args,
                          std::vector<AnfNodePtr> *inputs, ValuePtrList *weights_args);
  void MakeNestedCnode(bool has_custom_bprop, const std::vector<ValuePtr> &forward_args,
                       const FuncGraphPtr &cur_run_bprop_graph, const BaseRef &out);
  TopCellInfoPtr PopHighOrderGraphStack();
  void PushInputArgsInfoStack(const InputArgsInfoPtr &input_args_info);
  void PopInputArgsInfoStack();
  void HandleInputArgsForTopCell(const InputArgsInfoPtr &input_args_info, bool is_bprop_top) const;
  void InitResourceAndDfBuilder(const InputArgsInfoPtr &cell_info);
  void MakeNewTopGraph(const InputArgsInfoPtr &input_args_info);
  // Manage resource when run grad process.
  bool IsBpropGraph(const std::string &cell_id) const;
  void NewGraphInner(const py::object &obj, const py::args &args);
  void NewGraphImpl(const InputArgsInfoPtr &input_args_info);
  void SetForwardLastNodeInfo(const ValuePtr &v, const std::string &obj_id) const;
  void GetCustomBpropPrim(const py::object &obj, const py::args &args, const py::object &out,
                          const InputArgsInfoPtr &input_args_info);
  void DoGradForCustomBprop(const InputArgsInfoPtr &input_args_info, const std::string &out_id);
  void GetGradGraph(const ad::GradAttr &grad_attr, const std::vector<AnfNodePtr> &w_args,
                    const std::vector<size_t> &p_args);
  FuncGraphPtr GetBpropGraph(const ad::GradAttr &grad_attr, const vector<AnfNodePtr> &w_args,
                             const vector<size_t> &p_args);
  std::vector<AnfNodePtr> GetWeightsArgs(const py::object &weights, bool *weight_param_is_tuple) const;
  void CheckParamShapeAndType(const AnfNodePtr &param, const ParameterPtr &param_node,
                              const abstract::AbstractBasePtr &input_abs,
                              const abstract::AbstractBasePtr &param_tensor_abs, const std::string &input_shape);
  void UpdateParamAbsByArgs(const std::vector<ValuePtr> &input_args, const FuncGraphPtr &bprop_graph, bool has_sens);
  std::vector<size_t> GetGradPositionArgs(const py::object &grad_position, bool get_by_position) const;
  // Manage resource for construct forward graph.
  AnfNodePtr GetObjNode(const ValuePtr &v, const std::string &obj_id) const;
  AnfNodePtr CreateMakeTupleGradNode(const ValuePtr &v, const std::string &obj_id) const;
  AnfNodePtr CreateTupleGetItemNode(const std::string &obj_id) const;
  void RecordGradNodeToGraphInfoMap(const FuncGraphPtr &fg, const CNodePtr &cnode, const std::string &obj_id,
                                    const ValuePtrList &input_args) const;

  bool grad_flag_{false};
  bool grad_is_running_{false};
  bool need_renormalize_{false};
  bool eliminate_forward_{true};
  int custom_bprop_cell_count_{0};
  size_t cell_order_{0};
  // If grad_order=1, indicate first derivative; grad_order=2, indicate second derivative; ...
  size_t grad_order_{0};

  std::string grad_operation_;
  TopCellInfoPtr top_cell_{nullptr};
  InputArgsInfoPtr top_input_args_info_{nullptr};
  // Records every cell info for share, regardless of whether need construct grad graph
  std::stack<InputArgsInfoPtr> input_args_info_stack_;
  // For high grad of bprop
  std::stack<std::pair<std::string, bool>> bprop_grad_stack_;
  std::vector<std::string> bprop_cell_list_;
  // For high grad order
  std::stack<TopCellInfoPtr> high_order_stack_;
  ForwardExecutorWeakPtr forward_executor_;
  MsFunctionPtr ms_function_;
};
}  // namespace pynative
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_GRAD_H_
