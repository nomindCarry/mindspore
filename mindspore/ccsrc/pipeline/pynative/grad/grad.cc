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

#include "pipeline/pynative/grad/grad.h"
#include <algorithm>
#include "pipeline/pynative/grad/top_cell.h"
#include "pipeline/pynative/pynative_utils.h"
#include "pipeline/jit/pipeline.h"
#include "ir/cell.h"
#include "pipeline/jit/parse/data_converter.h"
#include "pipeline/jit/debug/trace.h"
#include "frontend/optimizer/ad/prim_bprop_optimizer.h"
#include "backend/common/optimizer/helper.h"
#include "include/common/utils/convert_utils_py.h"
#include "frontend/optimizer/ad/grad.h"
#include "pipeline/jit/pass.h"

namespace mindspore {
namespace pynative {
namespace {
const std::set<std::string> kHookOp = {"HookBackward", "CellBackwardHook"};
const char kGrad[] = "grad";

void SetGraphInputArgs(const std::vector<ValuePtr> &input_vec, const pipeline::ResourcePtr &res,
                       VectorRef *const arg_list) {
  MS_EXCEPTION_IF_NULL(arg_list);
  // Set inputs values
  for (auto v : input_vec) {
    (void)arg_list->emplace_back(v);
  }
  MS_EXCEPTION_IF_NULL(res);
  auto graph = res->func_graph();
  MS_EXCEPTION_IF_NULL(graph);
  std::vector<AnfNodePtr> graph_params = graph->parameters();
  std::size_t graph_params_size = graph_params.size();
  if ((*arg_list).size() != graph_params_size) {
    // Maybe have some default parameter for input
    for (std::size_t i = (*arg_list).size(); i < graph_params_size; ++i) {
      MS_EXCEPTION_IF_NULL(graph_params[i]);
      auto param_ptr = (graph_params[i])->cast_ptr<Parameter>();
      MS_EXCEPTION_IF_NULL(param_ptr);
      if (!param_ptr->has_default()) {
        MS_LOG(EXCEPTION) << "Parameter[" << i << "] has no default param";
      }
      if (!param_ptr->default_param()->isa<tensor::Tensor>()) {
        MS_LOG(EXCEPTION) << "Parameter[" << param_ptr->ToString()
                          << "] is not initialized, need to call `.init_data()`";
      }
      arg_list->push_back(param_ptr->default_param());
    }
  }
}

AbstractBasePtr GetGradGraphOutputAbstract(const FuncGraphPtr &fg) {
  MS_EXCEPTION_IF_NULL(fg);
  MS_EXCEPTION_IF_NULL(fg->output());
  return fg->output()->abstract();
}
}  // namespace

ForwardExecutorPtr GradExecutor::forward() const {
  auto forward_executor = forward_executor_.lock();
  MS_EXCEPTION_IF_NULL(forward_executor);
  return forward_executor;
}

std::string GradExecutor::GetCurCellOrder() const {
  if (input_args_info_stack_.empty()) {
    MS_LOG(EXCEPTION) << "The input_args_info_stack_ is empty!";
  }
  return input_args_info_stack_.top()->cell_id + "_" + std::to_string(cell_order_);
}

void GradExecutor::PushHighOrderGraphStack(const TopCellInfoPtr &top_cell) { high_order_stack_.push(top_cell); }

TopCellInfoPtr GradExecutor::PopHighOrderGraphStack() {
  if (high_order_stack_.empty()) {
    MS_LOG(EXCEPTION) << "Stack high_order_stack_ is empty";
  }
  high_order_stack_.pop();
  TopCellInfoPtr top_cell = nullptr;
  if (!high_order_stack_.empty()) {
    top_cell = high_order_stack_.top();
  }
  return top_cell;
}

void GradExecutor::PushInputArgsInfoStack(const InputArgsInfoPtr &input_args_info) {
  input_args_info_stack_.push(input_args_info);
  ++cell_order_;
}

void GradExecutor::PopInputArgsInfoStack() {
  if (input_args_info_stack_.empty()) {
    MS_LOG(EXCEPTION) << "Stack input_args_info_stack_ is empty";
  }
  input_args_info_stack_.pop();
}

inline bool GradExecutor::IsNestedGrad() const { return grad_order_ > 1; }

bool GradExecutor::IsBpropGraph(const std::string &cell_id) const {
  if (top_cell_ == nullptr) {
    return false;
  }
  return std::any_of(bprop_cell_list_.begin(), bprop_cell_list_.end(),
                     [&cell_id](const std::string &value) { return cell_id.find(value) != std::string::npos; });
}

void GradExecutor::HandleInputArgsForTopCell(const InputArgsInfoPtr &input_args_info, bool is_bprop_top) const {
  MS_EXCEPTION_IF_NULL(input_args_info);
  if (is_bprop_top) {
    // Convert input args to parameters for top cell graph in bprop.
    for (size_t i = 0; i < input_args_info->input_arg_id_vec.size(); ++i) {
      auto new_param = curr_g()->add_parameter();
      MS_LOG(DEBUG) << "Top bprop graph set input parameter " << input_args_info->input_arg_id_vec[i];
      top_cell()->SetParamNodeMapInGraphInfoMap(curr_g(), input_args_info->input_arg_id_vec[i], new_param);
    }
    return;
  }
  // Convert input args to parameters for top cell graph in construct.
  std::vector<ValuePtr> input_param_values;
  const auto &input_value = input_args_info->input_arg_value_vec;
  if (input_args_info->input_size != 0 && input_value.empty()) {
    MS_LOG(EXCEPTION) << "Input value is empty";
  }
  for (size_t i = 0; i < input_args_info->input_size; ++i) {
    const auto &v = input_value[i];
    if (!PyNativeAlgo::Common::IsTensor(v)) {
      continue;
    }
    auto new_param = curr_g()->add_parameter();
    (void)input_param_values.emplace_back(v);
    auto param_i_abs = v->ToAbstract();
    MS_EXCEPTION_IF_NULL(param_i_abs);
    param_i_abs = param_i_abs->Broaden();
    new_param->set_abstract(param_i_abs);
    top_cell()->SetParamNodeMapInGraphInfoMap(curr_g(), input_args_info->input_arg_id_vec[i], new_param, false);
  }
  top_cell()->set_k_pynative_cell_ptr(ad::GradPynativeCellBegin(curr_g()->parameters(), input_param_values));
}

void GradExecutor::InitResourceAndDfBuilder(const InputArgsInfoPtr &input_args_info) {
  MS_EXCEPTION_IF_NULL(input_args_info);
  if (input_args_info_stack_.empty() || IsNestedGrad()) {
    if (input_args_info_stack_.empty() && !grad_is_running_) {
      MS_LOG(DEBUG) << "Make new topest graph";
      MakeNewTopGraph(input_args_info);
    } else if (grad_is_running_ && IsBpropGraph(input_args_info->cell_id)) {
      MS_LOG(DEBUG) << "Run custom bprop cell";
      auto fg = std::make_shared<FuncGraph>();
      top_cell()->set_fg(fg);
      auto graph_info_cg = std::make_shared<GraphInfo>();
      top_cell()->SetGraphInfoMap(fg, graph_info_cg);
      HandleInputArgsForTopCell(input_args_info, true);
      bprop_grad_stack_.push(std::make_pair(input_args_info->cell_id, false));
    } else if (grad_is_running_ && top_cell()->grad_order() != grad_order_) {
      MS_LOG(DEBUG) << "Nested grad graph existed in custom bprop";
      MakeNewTopGraph(input_args_info);
      bprop_grad_stack_.push(std::make_pair(input_args_info->cell_id, true));
    } else if (!input_args_info_stack_.empty() && IsNestedGrad() && top_cell()->grad_order() != grad_order_) {
      MS_LOG(DEBUG) << "Nested grad graph existed in construct";
      MakeNewTopGraph(input_args_info);
    }
  }

  // Init kPynativeCellPtr with input parameters of top cell
  if (!top_cell()->is_init_kpynative()) {
    auto graph_info_cg = std::make_shared<GraphInfo>();
    top_cell()->SetGraphInfoMap(curr_g(), graph_info_cg);
    HandleInputArgsForTopCell(input_args_info, false);
    top_cell()->set_init_kpynative(true);
  }
}

void GradExecutor::NewGraphInner(const py::object &obj, const py::args &args) {
  bool is_top_grad_cell = input_args_info_stack_.empty();
  bool is_high_order_top_cell = IsNestedGrad() && top_cell()->grad_order() != grad_order_;
  const auto &input_args_info =
    PyNativeAlgo::PyParser::GetInputArgsInfo(obj, args, is_top_grad_cell || is_high_order_top_cell);
  // May be can async here
  NewGraphImpl(input_args_info);
}

void GradExecutor::NewGraphImpl(const InputArgsInfoPtr &input_args_info) {
  MS_EXCEPTION_IF_NULL(input_args_info);
  const auto &cell_id = input_args_info->cell_id;
  MS_LOG(DEBUG) << "NewGraphInner start " << input_args_info->input_size << ", cell_id " << cell_id
                << ", input args info ptr " << input_args_info.get();
  // When the cell has custom bprop, in_custom_bprop_cell is lager than 0
  if (input_args_info->has_custom_bprop) {
    custom_bprop_cell_count_ += 1;
  }
  // Make top graph and init resource
  InitResourceAndDfBuilder(input_args_info);
  PushInputArgsInfoStack(input_args_info);
}

void GradExecutor::MakeNewTopGraph(const InputArgsInfoPtr &input_args_info) {
  MS_EXCEPTION_IF_NULL(input_args_info);
  // Run forward first need plus 1
  if (grad_order_ == 0) {
    ++grad_order_;
  }
  auto fg = std::make_shared<FuncGraph>();
  auto resource = std::make_shared<pipeline::Resource>();
  const auto &already_run_cell_id = GetAlreadyRunCellId(input_args_info->cell_id);
  top_cell_ = std::make_shared<TopCellInfo>(grad_order_, input_args_info->cell_id, already_run_cell_id, resource, fg);
  top_cell_->set_input_args_info(input_args_info);
  top_cell_->set_forward_already_run(true);
  top_cell_->set_input_args_id(input_args_info->input_args_id);
  PushHighOrderGraphStack(top_cell_);
  MS_LOG(DEBUG) << "New top graph, fg ptr " << fg.get() << " resource ptr " << resource.get();
}

void GradExecutor::SetForwardLastNodeInfo(const ValuePtr &v, const std::string &obj_id) const {
  MS_EXCEPTION_IF_NULL(v);
  auto output_node = GetObjNode(v, obj_id);
  if (v->isa<tensor::CSRTensor>()) {
    auto csr_tensorptr = v->cast<tensor::CSRTensorPtr>();
    auto value_ptr = csr_tensorptr->GetValues();
    output_node = GetObjNode(value_ptr, PyNativeAlgo::Common::GetIdByValue(value_ptr));
  } else if (v->isa<tensor::COOTensor>()) {
    auto coo_tensorptr = v->cast<tensor::COOTensorPtr>();
    auto value_ptr = coo_tensorptr->GetValues();
    output_node = GetObjNode(value_ptr, PyNativeAlgo::Common::GetIdByValue(value_ptr));
  }
  MS_EXCEPTION_IF_NULL(output_node);
  if (output_node->abstract() == nullptr) {
    output_node->set_abstract(v->ToAbstract()->Broaden());
  }
  // Set last output abstract and will be used for sens
  auto k_pynative_cell_ptr = top_cell()->k_pynative_cell_ptr();
  MS_EXCEPTION_IF_NULL(k_pynative_cell_ptr);
  k_pynative_cell_ptr->UpdateOutputNodeOfTopCell(output_node, v);
}

void GradExecutor::EndGraphInner(const py::object &obj, const py::object &out, const py::args &args) {
  if (input_args_info_stack_.empty()) {
    return;
  }
  const auto input_args_info = input_args_info_stack_.top();
  MS_EXCEPTION_IF_NULL(input_args_info);
  if (input_args_info->has_custom_bprop) {
    GetCustomBpropPrim(obj, args, out, input_args_info);
  }
  input_args_info->out_value = PyNativeAlgo::DataConvert::PyObjToValue(out);
  // May be can async here
  EndGraphImpl(input_args_info);
}

void GradExecutor::EndGraphImpl(const InputArgsInfoPtr &input_args_info) {
  MS_EXCEPTION_IF_NULL(input_args_info);
  const auto &cell_id = input_args_info->cell_id;
  MS_LOG(DEBUG) << "EndGraphInner start " << input_args_info->input_size << ", cell_id " << cell_id
                << ", input args info ptr " << input_args_info.get();
  const auto &out_value = input_args_info->out_value;
  MS_EXCEPTION_IF_NULL(out_value);
  const auto &out_id = PyNativeAlgo::Common::GetIdByValue(out_value);
  DoGradForCustomBprop(input_args_info, out_id);
  PopInputArgsInfoStack();
  // Update bprop grad stack
  if (grad_is_running_ && !bprop_grad_stack_.empty()) {
    if (!bprop_grad_stack_.top().second) {
      curr_g()->set_output(GetObjNode(input_args_info->out_value, out_id));
      bprop_grad_stack_.pop();
      return;
    } else if (bprop_grad_stack_.top().first == input_args_info->cell_id) {
      bprop_grad_stack_.pop();
    }
  }
  // Just only dump the last forward graph
  bool is_top_cell_end = cell_id == top_cell()->cell_id();
  if (is_top_cell_end && MsContext::GetInstance()->get_param<bool>(MS_CTX_SAVE_GRAPHS_FLAG)) {
    curr_g()->set_output(GetObjNode(out_value, out_id));
    PyNativeAlgo::Common::DumpGraphIR("fg.ir", curr_g());
  }
  // Reset grad flag and update output node of the outermost cell
  if (input_args_info_stack_.empty() && is_top_cell_end) {
    MS_LOG(DEBUG) << "Cur top last cell " << cell_id;
    (void)PopHighOrderGraphStack();
    SetForwardLastNodeInfo(out_value, out_id);
    top_cell()->ClearCellHookOp();
    cell_order_ = 0;
    set_grad_flag(false);
  }
  // Checkout whether need to compile graph when each top cell has ran finished
  if (is_top_cell_end) {
    // In high grad cases, the output of the internal graph may be a tuple, and node needs to be created in the getobj
    if (!input_args_info_stack_.empty()) {
      SetForwardLastNodeInfo(out_value, out_id);
    }
    top_cell()->CheckSubCellHookChanged();
    top_input_args_info_ = input_args_info;
  }
}

void GradExecutor::DoGradForCustomBprop(const InputArgsInfoPtr &input_args_info, const std::string &out_id) {
  MS_EXCEPTION_IF_NULL(input_args_info);
  if (!input_args_info->has_custom_bprop || custom_bprop_cell_count_ != 0) {
    return;
  }
  MS_LOG(DEBUG) << "Do grad for custom bprop";
  MS_EXCEPTION_IF_NULL(input_args_info->custom_bprp_prim);
  auto op_run_info = std::make_shared<FrontendOpRunInfo>();
  op_run_info->base_op_run_info.op_name = input_args_info->custom_bprp_prim->name();
  op_run_info->op_prim = input_args_info->custom_bprp_prim;
  op_run_info->input_value = input_args_info->input_arg_value_vec;
  op_run_info->input_size = input_args_info->input_arg_value_vec.size();
  op_run_info->input_value_id = input_args_info->input_arg_id_vec;
  auto cnode = ConstructForwardGraph(op_run_info);
  DoOpGrad(op_run_info, cnode, input_args_info->out_value);
  SaveOutputNodeMap(out_id, op_run_info, cnode);
}

void GradExecutor::GetCustomBpropPrim(const py::object &obj, const py::args &args, const py::object &out,
                                      const InputArgsInfoPtr &input_args_info) {
  custom_bprop_cell_count_ -= 1;
  if (custom_bprop_cell_count_ != 0) {
    return;
  }
  MS_LOG(DEBUG) << "Get custom bprop prim";
  py::function bprop_func = py::getattr(obj, parse::CUSTOM_BPROP_NAME);
  py::object code_obj = py::getattr(bprop_func, "__code__");
  // When the co_names is empty, we will still get a tuple which is empty.
  auto co_names = py::getattr(code_obj, "co_names").cast<py::tuple>();
  for (auto name : co_names) {
    if (!py::hasattr(obj, name)) {
      continue;
    }
    auto var = py::getattr(obj, name);
    if (py::hasattr(var, "__parameter__") && py::isinstance<tensor::MetaTensor>(var)) {
      MS_LOG(EXCEPTION) << "The user defined 'bprop' function does not support using Parameter.";
    }
  }

  py::object co_name = py::getattr(code_obj, "co_name");
  if (std::string(py::str(co_name)) == "staging_specialize") {
    MS_LOG(EXCEPTION) << "Decorating bprop with '@jit' is not supported.";
  }
  // Three parameters self, out and dout need to be excluded
  const size_t inputs_num = static_cast<size_t>(py::getattr(code_obj, "co_argcount").cast<int64_t>() - 3);
  if (inputs_num != args.size()) {
    MS_EXCEPTION(TypeError) << "Size of bprop func inputs[" << inputs_num
                            << "] is not equal to the size of cell inputs[" << args.size() << "]";
  }

  auto bprop_func_cellid = PyNativeAlgo::PyParser::GetIdByPyObj(bprop_func);
  (void)bprop_cell_list_.emplace_back(bprop_func_cellid);
  auto fake_prim = std::make_shared<PrimitivePy>(prim::kPrimHookBackward->name());
  MS_EXCEPTION_IF_NULL(input_args_info);
  if (py::isinstance<Cell>(obj)) {
    const auto &cell_ptr = obj.cast<CellPtr>();
    fake_prim->set_bprop_cls_name(cell_ptr->name());
  }
  fake_prim->AddBackwardHookFn(0, bprop_func);

  (void)fake_prim->AddAttr("cell_id", MakeValue(input_args_info->cell_id));
  (void)fake_prim->AddAttr(parse::CUSTOM_BPROP_NAME, MakeValue(true));
  if (input_args_info->input_arg_value_vec.empty()) {
    for (size_t i = 0; i < args.size(); ++i) {
      (void)input_args_info->input_arg_value_vec.emplace_back(PyNativeAlgo::DataConvert::PyObjToValue(args[i]));
    }
  }
  input_args_info->custom_bprp_prim = fake_prim;
}

std::string GradExecutor::GetAlreadyRunCellId(const std::string &cell_id) const {
  std::string already_run_cell_id(cell_id);
  already_run_cell_id += std::to_string(grad_order_ == 0 ? 1 : grad_order_);
  already_run_cell_id += "_" + grad_operation_;
  MS_LOG(DEBUG) << "Get already run top cell id " << already_run_cell_id;
  return already_run_cell_id;
}

void GradExecutor::GradNetInner(const prim::GradOperationPtr &grad, const py::object &obj, const py::object &weights,
                                const py::object &grad_position, const py::args &args) {
  MS_EXCEPTION_IF_NULL(grad);
  MS_EXCEPTION_IF_NULL(top_input_args_info_);
  MS_LOG(DEBUG) << "GradNetInner start " << args.size() << ", cell_id " << top_input_args_info_->cell_id
                << ", input args info ptr " << top_input_args_info_.get();
  if (grad->sens_param()) {
    MS_LOG(DEBUG) << "Get sens param";
    size_t forward_args_size = args.size() - 1;
    const auto &sens_v = PyNativeAlgo::DataConvert::PyObjToValue(args[forward_args_size]);
    // Sens have already exist, which may be need update
    if (top_input_args_info_->input_arg_value_vec.size() == args.size()) {
      top_input_args_info_->input_arg_value_vec.pop_back();
    }
    (void)top_input_args_info_->input_arg_value_vec.emplace_back(ShallowCopyTensorValue(sens_v));
  }

  MS_LOG(DEBUG) << "Need compile graph";
  SetBpropGraphJitLevel(obj);
  top_cell()->set_grad_operation(grad_operation_);
  bool weight_param_is_tuple = true;
  auto w_args = GetWeightsArgs(weights, &weight_param_is_tuple);
  auto p_args = GetGradPositionArgs(grad_position, grad->get_by_position_);
  ad::GradAttr grad_attr(grad->get_all_, grad->get_by_list_, grad->sens_param_, grad->get_by_position_,
                         weight_param_is_tuple);
  GetGradGraph(grad_attr, w_args, p_args);
}

void GradExecutor::GetGradGraph(const ad::GradAttr &grad_attr, const std::vector<AnfNodePtr> &w_args,
                                const std::vector<size_t> &p_args) {
  // Get bprop graph of top cell
  auto bprop_graph = GetBpropGraph(grad_attr, w_args, p_args);
  MS_EXCEPTION_IF_NULL(bprop_graph);
  bprop_graph->set_flag(kFlagIsPynativeBpropGraph, true);
  auto resource = top_cell()->resource();
  MS_EXCEPTION_IF_NULL(resource);
  resource->set_func_graph(bprop_graph);
  auto manager = resource->manager();
  MS_EXCEPTION_IF_NULL(manager);
  manager->AddFuncGraph(bprop_graph, true);
  PyNativeAlgo::Common::DumpGraphIR("launch_bprop_graph.ir", bprop_graph);
  compile::SetMindRTEnable();
  resource->SetBackendAsync([]() { return compile::CreateBackend(); });
  MS_LOG(DEBUG) << "Start task emit action";
  (void)TaskEmitAction(resource);
  MS_LOG(DEBUG) << "Start execute action";
  (void)ExecuteAction(resource);
}

std::vector<AnfNodePtr> GradExecutor::GetWeightsArgs(const py::object &weights, bool *weight_param_is_tuple) const {
  std::vector<AnfNodePtr> w_args;
  if (py::hasattr(weights, "__parameter_tuple__")) {
    const auto &tuple = weights.cast<py::tuple>();
    MS_LOG(DEBUG) << "Get weights tuple size " << tuple.size();
    for (size_t i = 0; i < tuple.size(); ++i) {
      const auto &v = PyNativeAlgo::DataConvert::PyObjToValue(tuple[i]);
      const auto &param = GetParamInput(v, PyNativeAlgo::Common::GetIdByValue(v));
      MS_EXCEPTION_IF_NULL(param);
      w_args.emplace_back(param);
    }
  } else {
    MS_LOG(DEBUG) << "No parameter tuple get, add weights params by input weight";
    if (py::isinstance<py::tuple>(weights) || py::isinstance<py::list>(weights)) {
      auto weights_list = py::cast<py::list>(weights);
      for (size_t i = 0; i < weights_list.size(); ++i) {
        (void)w_args.emplace_back(GetInput(PyNativeAlgo::DataConvert::PyObjToValue(weights_list[i])));
      }
    } else if (!py::isinstance<py::none>(weights)) {
      // Single input
      (void)w_args.emplace_back(GetInput(PyNativeAlgo::DataConvert::PyObjToValue(weights)));
      *weight_param_is_tuple = false;
    }
  }
  return w_args;
}

std::vector<size_t> GradExecutor::GetGradPositionArgs(const py::object &grad_position, bool get_by_position) const {
  std::vector<size_t> pos_args;
  if (!get_by_position) {
    return pos_args;
  }
  if (py::isinstance<py::tuple>(grad_position)) {
    const auto &tuple = grad_position.cast<py::tuple>();
    (void)std::transform(tuple.begin(), tuple.end(), std::back_inserter(pos_args),
                         [](const py::handle &elem) { return elem.cast<int64_t>(); });
    return pos_args;
  }
  MS_LOG(EXCEPTION) << "Grad position only support tuple when grad_by_position is set True.";
}

void GradExecutor::CheckParamShapeAndType(const AnfNodePtr &param, const ParameterPtr &param_node,
                                          const abstract::AbstractBasePtr &input_abs,
                                          const abstract::AbstractBasePtr &param_tensor_abs,
                                          const std::string &input_shape) {
  MS_EXCEPTION_IF_NULL(param);
  MS_EXCEPTION_IF_NULL(param_node);
  MS_EXCEPTION_IF_NULL(param_tensor_abs);
  auto ir_base_shape = param_tensor_abs->BuildShape();
  MS_EXCEPTION_IF_NULL(ir_base_shape);
  auto ir_shape = ir_base_shape->ToString();
  if (input_shape != "()" && ir_shape != "()") {
    if (input_shape != ir_shape) {
      // Sens shape in ir graph is determined by graph output, so it can be dynamic shape; But input shape is
      // determined by user input, which could not be dynamic shape.
      if (param_node->debug_info()->name() != "sens" || !ir_base_shape->IsDynamic()) {
        MS_EXCEPTION(ValueError) << "The shape should be " << ir_shape << ", but got " << input_shape << ", "
                                 << param->DebugString();
      }
    }
    auto ir_dtype = param_tensor_abs->BuildType()->ToString();
    MS_EXCEPTION_IF_NULL(input_abs);
    auto input_dtype = input_abs->BuildType()->ToString();
    if (input_dtype != ir_dtype) {
      MS_EXCEPTION(TypeError) << "The dtype should be " << ir_dtype << ", but got " << input_dtype << ", "
                              << param->DebugString();
    }
  }
  if (param_node->debug_info()->name() == "sens" && ir_shape != input_shape) {
    need_renormalize_ = true;
  }
}

void GradExecutor::UpdateParamAbsByArgs(const std::vector<ValuePtr> &input_args, const FuncGraphPtr &bprop_graph,
                                        bool has_sens) {
  MS_EXCEPTION_IF_NULL(bprop_graph);
  std::vector<ValuePtr> tensor_args;
  size_t input_size = has_sens ? input_args.size() - 1 : input_args.size();
  // Sens may be a value tuple not a single tensor
  for (size_t i = 0; i < input_size; ++i) {
    if (PyNativeAlgo::Common::IsTensor(input_args[i])) {
      (void)tensor_args.emplace_back(input_args[i]);
    }
  }
  if (has_sens) {
    (void)tensor_args.emplace_back(input_args[input_size]);
  }
  const auto &bprop_params = bprop_graph->parameters();
  // bprop_params include inputs, parameters and sens, should be more than inputs size
  if (bprop_params.size() < tensor_args.size()) {
    MS_LOG(EXCEPTION) << "Df parameters size " << bprop_params.size() << " less than " << tensor_args.size();
  }
  size_t index = 0;
  for (const auto &param : bprop_params) {
    auto param_node = param->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(param_node);
    if (param_node->has_default()) {
      // update abstract info for weights
      ValuePtr value = param_node->default_param();
      MS_EXCEPTION_IF_NULL(value);
      auto ptr = value->ToAbstract();
      MS_EXCEPTION_IF_NULL(ptr);
      param_node->set_abstract(ptr->Broaden());
    } else {
      // Update abstract info for input params
      auto input_abs = abstract::FromValue(tensor_args[index], true);
      MS_EXCEPTION_IF_NULL(input_abs);
      if (param_node->abstract() != nullptr) {
        auto input_shape = input_abs->BuildShape()->ToString();
        auto param_tensor_abs = param_node->abstract();
        if (param_tensor_abs->isa<abstract::AbstractRefTensor>()) {
          param_tensor_abs = param_tensor_abs->cast<abstract::AbstractRefPtr>()->CloneAsTensor();
        }
        CheckParamShapeAndType(param, param_node, input_abs, param_tensor_abs, input_shape);
      }
      param_node->set_abstract(input_abs->Broaden());
      index++;
    }
  }
}

FuncGraphPtr GradExecutor::GetBpropGraph(const ad::GradAttr &grad_attr, const vector<AnfNodePtr> &w_args,
                                         const vector<size_t> &p_args) {
  MS_EXCEPTION_IF_NULL(top_input_args_info_);
  bool build_formal_param = false;
  if (!top_input_args_info_->has_custom_bprop && !input_args_info_stack_.empty() && IsNestedGrad()) {
    build_formal_param = true;
    need_renormalize_ = true;
  }
  if (top_cell()->ms_function_flag()) {
    need_renormalize_ = true;
  }

  auto k_pynative_cell_ptr = top_cell()->k_pynative_cell_ptr();
  MS_EXCEPTION_IF_NULL(k_pynative_cell_ptr);
  FuncGraphPtr bprop_graph =
    ad::GradPynativeCellEnd(k_pynative_cell_ptr, w_args, p_args, grad_attr, build_formal_param);
  MS_EXCEPTION_IF_NULL(bprop_graph);

  MS_LOG(DEBUG) << "Top graph input params size " << top_input_args_info_->input_arg_value_vec.size();
  std::ostringstream ss;
  ss << "grad{" << top_input_args_info_->input_arg_value_vec.size() << "}";
  bprop_graph->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  bprop_graph->debug_info()->set_name(ss.str());
  UpdateParamAbsByArgs(top_input_args_info_->input_arg_value_vec, bprop_graph, grad_attr.has_sens);
  // Do opt for final bprop graph
  pipeline::ResourcePtr resource = std::make_shared<pipeline::Resource>();
  resource->set_func_graph(bprop_graph);
  auto manager = resource->manager();
  MS_EXCEPTION_IF_NULL(manager);
  manager->AddFuncGraph(bprop_graph);
  auto optimized_bg = ad::PrimBpropOptimizer::GetPrimBpropOptimizerInst().BpropGraphFinalOpt(resource);
  if (input_args_info_stack_.empty()) {
    need_renormalize_ = false;
  }
  PyNativeAlgo::Common::DumpGraphIR("after_final_opt.ir", optimized_bg);
  return optimized_bg;
}

void GradExecutor::SetGradOrder(const py::object &obj, const py::args &args, const std::string &cell_id) {
  // top_cell_ == nullptr means call by grad api
  // cell_id.find(top_cell_->cell_id()) == std::string::npos means current cell is not top cell, may be high order
  if (top_cell_ == nullptr || cell_id.find(top_cell_->cell_id()) == std::string::npos) {
    ++grad_order_;
  }
  if (!grad_is_running_) {
    MS_LOG(DEBUG) << "Grad not running yet";
    return;
  }
}

py::object GradExecutor::CheckAlreadyRun(const prim::GradOperationPtr &grad, const py::object &obj,
                                         const py::object &grad_hash_id, const py::args &args) {
  const auto &cell_id = PyNativeAlgo::PyParser::GetCellId(obj, args);
  // Check current cell grad order and erase it if in current top cell list
  SetGradOrder(obj, args, cell_id);

  bool forward_run = false;
  // Get cell id and input args info
  std::string grad_hash_id_str;
  if (!py::isinstance<py::none>(grad_hash_id)) {
    grad_hash_id_str = std::string(py::str(grad_hash_id));
  }
  grad_operation_ = std::to_string(static_cast<int>(grad->get_all_)) +
                    std::to_string(static_cast<int>(grad->get_by_list_)) + grad_hash_id_str;

  std::string input_args_id;
  for (size_t i = 0; i < args.size(); ++i) {
    input_args_id += PyNativeAlgo::PyParser::GetIdByPyObj(args[i]) + "_";
  }
  // Under the condition that the stack is empty (forward process completed or no forward process),
  // check whether need to run forward process
  if (input_args_info_stack_.empty() && top_cell_ != nullptr) {
    const auto &check_already_run_cell_id = GetAlreadyRunCellId(cell_id);
    auto find_top_cell = GetTopCell(check_already_run_cell_id);
    if (find_top_cell != nullptr) {
      MS_LOG(DEBUG) << "Find already run top cell";
      forward_run = find_top_cell->forward_already_run();
      bool input_args_changed =
        !find_top_cell->input_args_id().empty() && find_top_cell->input_args_id() != input_args_id;
      if (forward_run && input_args_changed) {
        MS_LOG(WARNING) << "The construct of running cell is dynamic and the input info of this cell has changed, "
                           "forward process will run again";
        forward_run = false;
      }
    }
  }
  MS_LOG(DEBUG) << "Graph have already ran " << forward_run << " top cell id " << cell_id;
  return BaseRefToPyData(forward_run);
}

py::object GradExecutor::RunGradGraph() {
  MS_EXCEPTION_IF_NULL(top_input_args_info_);
  const auto &resource = top_cell()->resource();
  MS_EXCEPTION_IF_NULL(resource);
  MS_LOG(DEBUG) << "Run cell id " << top_input_args_info_->cell_id << ", resource ptr " << resource.get();
  std::vector<ValuePtr> flatten_v;
  PyNativeAlgo::DataConvert::FlattenArgs(top_input_args_info_->input_arg_value_vec, &flatten_v);
  VectorRef arg_list;
  SetGraphInputArgs(flatten_v, resource, &arg_list);
  MS_LOG(DEBUG) << "Convert args size " << flatten_v.size() << ", graph param size " << arg_list.size();
  compile::VmEvalFuncPtr run = resource->GetResult(pipeline::kOutput).cast<compile::VmEvalFuncPtr>();
  MS_EXCEPTION_IF_NULL(run);

  const auto &backend = MsContext::GetInstance()->backend_policy();
  MS_LOG(DEBUG) << "Eval run " << backend;
  grad_is_running_ = true;
  top_cell()->set_k_pynative_cell_ptr(nullptr);
  BaseRef out_value = (*run)(arg_list);
  grad_is_running_ = false;
  MS_LOG(DEBUG) << "Eval run end " << out_value.ToString();
  const auto &cur_run_bprop_graph = resource->func_graph();
  const auto &out_abs = GetGradGraphOutputAbstract(cur_run_bprop_graph);
  MakeNestedCnode(top_input_args_info_->has_custom_bprop, flatten_v, cur_run_bprop_graph, out_value);
  return BaseRefToPyData(out_value, out_abs);
}

void GradExecutor::MakeNestedCnode(bool has_custom_bprop, const std::vector<ValuePtr> &forward_args,
                                   const FuncGraphPtr &cur_run_bprop_graph, const BaseRef &out) {
  if (input_args_info_stack_.empty()) {
    MS_LOG(DEBUG) << "No nested grad find";
    // Custom bprop nested, top cell reset by first time, second time no need clean
    if (top_cell_ != nullptr) {
      top_cell_->ClearDeviceMemory();
    }
    ClearTopCellRes();
    ClearGlobalRes();
    return;
  }
  FuncGraphPtr first_grad_fg = cur_run_bprop_graph;
  if (has_custom_bprop) {
    first_grad_fg = curr_g();
    MS_LOG(DEBUG) << "Bprop nested";
  }
  MS_EXCEPTION_IF_NULL(first_grad_fg);
  PyNativeAlgo::Common::DumpGraphIR("first_grad_fg.ir", first_grad_fg);

  std::vector<AnfNodePtr> inputs{NewValueNode(first_grad_fg)};
  ValuePtrList weights_args;
  DoParameterReplace(first_grad_fg, forward_args, &inputs, &weights_args);

  pipeline::ResourcePtr r = std::make_shared<pipeline::Resource>();
  r->manager()->AddFuncGraph(first_grad_fg);
  set_eliminate_forward(false);
  (void)first_grad_fg->transforms().erase(kGrad);
  // Do high order
  FuncGraphPtr second_grad_fg = ad::Grad(first_grad_fg, opt::Optimizer::MakeEmptyOptimizer(r));
  set_eliminate_forward(true);
  PyNativeAlgo::Common::DumpGraphIR("second_grad_fg.ir", second_grad_fg);
  r->Clean();

  MS_LOG(DEBUG) << "Get cur graph ptr " << curr_g().get();
  auto cnode = curr_g()->NewCNode(inputs);
  auto out_value = PyNativeAlgo::DataConvert::BaseRefToValue(out);
  const auto &out_id = PyNativeAlgo::Common::GetIdByValue(out_value);
  top_cell()->SetNodeMapInGraphInfoMap(curr_g(), out_id, cnode);
  MS_LOG(DEBUG) << "Nested make cnode is " << cnode->DebugString();

  // Get input values
  ValuePtrList input_args(forward_args);
  (void)input_args.insert(input_args.end(), weights_args.cbegin(), weights_args.cend());
  // Get output values
  if (has_custom_bprop && !out_value->isa<ValueSequence>()) {
    std::vector<ValuePtr> out_v{out_value};
    out_value = std::make_shared<ValueTuple>(out_v);
  }
  if (!top_cell()->k_pynative_cell_ptr()->KPynativeWithFProp(cnode, input_args, out_value, second_grad_fg)) {
    MS_LOG(EXCEPTION) << "Failed to run ad grad for second grad graph " << cnode->ToString();
  }
  need_renormalize_ = true;
}

void GradExecutor::DoParameterReplace(const FuncGraphPtr &first_grad_fg, const std::vector<ValuePtr> &forward_args,
                                      std::vector<AnfNodePtr> *inputs, ValuePtrList *weights_args) {
  auto inner_graph_info = top_cell()->graph_info_map().at(curr_g());
  MS_EXCEPTION_IF_NULL(inner_graph_info);
  // Change current top cell to outer top cell
  SwitchTopCell();
  auto outter_graph_info = top_cell()->graph_info_map().at(curr_g());
  MS_EXCEPTION_IF_NULL(outter_graph_info);

  auto manager = Manage({first_grad_fg}, false);
  // Replace inputs param
  MS_EXCEPTION_IF_NULL(inputs);
  for (size_t i = 0; i < forward_args.size(); ++i) {
    const auto &id = PyNativeAlgo::Common::GetIdByValue(forward_args[i]);
    if (outter_graph_info->input_params.find(id) != outter_graph_info->input_params.end()) {
      // Can find in second graph
      MS_LOG(DEBUG) << "Replace input param id " << id;
      const auto &input_param_second = outter_graph_info->input_params.at(id);
      // Replace inner graph param by outter outter graph param
      (void)manager->Replace(inner_graph_info->input_params.at(id), input_param_second);
      (void)inputs->emplace_back(input_param_second);
    } else {
      MS_LOG(DEBUG) << "Can't find input param id " << id << " in outer graph";
      // Inner graph input param not find in outter graph, need add to outter graph
      (void)inputs->emplace_back(GetInput(forward_args[i]));
    }
  }

  // Replace weights param
  MS_EXCEPTION_IF_NULL(weights_args);
  for (const auto &weight : inner_graph_info->weight_params) {
    if (outter_graph_info->weight_params.find(weight.first) != outter_graph_info->weight_params.end()) {
      // Can find in second graph
      MS_LOG(DEBUG) << "Replace weight param name " << weight.first << " ptr " << weight.second.get();
      const auto it = std::find_if(outter_graph_info->weight_params.begin(), outter_graph_info->weight_params.begin(),
                                   [&weight](const std::pair<std::string, ParameterPtr> &sec) {
                                     return weight.second->name() == sec.second->name();
                                   });
      if (it != outter_graph_info->weight_params.end()) {
        (void)manager->Replace(weight.second, it->second);
        (void)inputs->emplace_back(it->second);
        (void)weights_args->emplace_back(it->second->default_param());
      }
    } else {
      MS_LOG(DEBUG) << "Can't find weight param " << weight.first << " in outer graph";
      curr_g()->add_parameter(weight.second);
      top_cell()->SetParamNodeMapInGraphInfoMap(curr_g(), weight.first, weight.second, true);
      (void)inputs->emplace_back(weight.second);
      (void)weights_args->emplace_back(weight.second->default_param());
    }
  }
}

void GradExecutor::SwitchTopCell() {
  // Clear current top cell res
  ClearTopCellRes();
  // Get outer top cell
  auto outer_top_cell = PopHighOrderGraphStack();
  MS_EXCEPTION_IF_NULL(outer_top_cell);
  set_top_cell(outer_top_cell);
}

void GradExecutor::ClearGlobalRes() {
  abstract::AnalysisContext::ClearContext();
  parse::data_converter::ClearObjectCache();
  parse::Parser::CleanParserResource();
  trace::ClearTraceStack();
  ad::CleanRes();
  pipeline::ReclaimOptimizer();
}

void GradExecutor::ClearRes() {
  MS_LOG(DEBUG) << "Clear grad res";
  grad_flag_ = false;
  grad_is_running_ = false;
  need_renormalize_ = false;
  eliminate_forward_ = true;
  custom_bprop_cell_count_ = 0;
  grad_order_ = 0;
  grad_operation_.clear();
  top_cell_ = nullptr;
  top_input_args_info_ = nullptr;
  bprop_cell_list_.clear();
  std::stack<InputArgsInfoPtr>().swap(input_args_info_stack_);
  std::stack<std::pair<std::string, bool>>().swap(bprop_grad_stack_);
  std::stack<TopCellInfoPtr>().swap(high_order_stack_);
}

void GradExecutor::ClearTopCellRes() {
  if (grad_order_ > 0) {
    --grad_order_;
  }
  // Clear current top cell
  top_cell_ = nullptr;
}

AnfNodePtr GradExecutor::GetInput(const ValuePtr &v) const {
  MS_EXCEPTION_IF_NULL(v);
  const auto &obj_id = PyNativeAlgo::Common::GetIdByValue(v);
  const auto &fg = top_cell()->fg();

  // Get param input
  AnfNodePtr node = GetParamInput(v, obj_id);
  if (node != nullptr) {
    MS_LOG(DEBUG) << "Get input node " << node->ToString() << ", id " << obj_id;
    return node;
  }

  // Get op output
  auto curr_graph_info = top_cell()->graph_info_map().at(fg);
  MS_EXCEPTION_IF_NULL(curr_graph_info);
  if (curr_graph_info->node_map.find(obj_id) != curr_graph_info->node_map.end()) {
    // op(x, y)
    // out = op(op1(x, y))
    // out = op(cell1(x, y))
    // out = op(cell1(x, y)[0])
    node = GetObjNode(v, obj_id);
  } else if (v->isa<ValueSequence>()) {
    // out = op((x, y))
    // out = cell((x, y))
    auto tuple = v->cast<ValueSequencePtr>();
    // cell((1,2)): support not mix (scalar, tensor)
    if (tuple->size() != 0 && !tuple->value()[0]->isa<tensor::Tensor>()) {
      node = NewValueNode(v);
    } else {
      std::vector<AnfNodePtr> args;
      (void)args.emplace_back(NewValueNode(prim::kPrimMakeTuple));
      auto tuple_size = tuple->size();
      for (size_t i = 0; i < tuple_size; i++) {
        (void)args.emplace_back(GetInput(tuple->value()[i]));
      }
      node = fg->NewCNode(args);
    }
  } else {
    node = NewValueNode(v);
  }
  node == nullptr ? MS_LOG(DEBUG) << "Get node is nullptr"
                  : MS_LOG(DEBUG) << "Get input node " << node->ToString() << ", id " << obj_id;
  return node;
}

AnfNodePtr GradExecutor::GetParamInput(const ValuePtr &v, const std::string &id) const {
  const auto &graph_info = top_cell()->graph_info_map().at(curr_g());
  MS_EXCEPTION_IF_NULL(graph_info);
  // Get input param input
  const auto it = graph_info->input_params.find(id);
  if (it != graph_info->input_params.end()) {
    return it->second;
  }

  // Get weight param input
  if (v->isa<tensor::Tensor>() && v->cast<tensor::TensorPtr>()->is_parameter()) {
    MS_LOG(DEBUG) << "Weight param";
    const auto item_by_id = graph_info->weight_params.find(id);
    if (item_by_id != graph_info->weight_params.end()) {
      return item_by_id->second;
    }
    const auto &tensor = v->cast<tensor::TensorPtr>();
    const auto &param_info = tensor->param_info();
    MS_EXCEPTION_IF_NULL(param_info);
    const auto &param_name = param_info->name();
    const auto item_by_name = graph_info->weight_params.find(param_name);
    // In ms function, GetWeightsNode save to graph info by weight param name
    if (item_by_name != graph_info->weight_params.end()) {
      return item_by_name->second;
    }
    // Add new weight param to graph info
    auto weight_param = curr_g()->add_parameter();
    weight_param->set_name(param_name);
    weight_param->debug_info()->set_name(param_name);
    weight_param->set_default_param(tensor);
    top_cell()->SetParamNodeMapInGraphInfoMap(top_cell()->fg(), id, weight_param, true);
    return weight_param;
  }
  return nullptr;
}

AnfNodePtr GradExecutor::GetObjNode(const ValuePtr &v, const std::string &obj_id) const {
  MS_EXCEPTION_IF_NULL(v);
  const auto &fg = top_cell()->fg();
  auto graph_info = top_cell()->graph_info_map().at(fg);
  MS_EXCEPTION_IF_NULL(graph_info);
  if (graph_info->node_map.find(obj_id) == graph_info->node_map.end()) {
    // A tuple returns in this case: x = op1, y = op2, return (x, y)
    // or a constant returns in this case
    auto make_tuple = CreateMakeTupleGradNode(v, obj_id);
    if (make_tuple == nullptr) {
      MS_LOG(DEBUG) << "Create value node for obj id: " << obj_id;
      return NewValueNode(v);
    }
    return make_tuple;
  }
  // Single output CNode
  const auto &out = graph_info->node_map.at(obj_id);
  if (out.second.size() == 1 && out.second[0] == -1) {
    return out.first;
  }
  // Create tuple get item node for multiple output CNode
  return CreateTupleGetItemNode(obj_id);
}

void GradExecutor::RecordGradNodeToGraphInfoMap(const FuncGraphPtr &fg, const CNodePtr &cnode,
                                                const std::string &obj_id, const ValuePtrList &input_args) const {
  top_cell()->SetNodeMapInGraphInfoMap(fg, obj_id, cnode);
  // run ad for make tuple node
  if (grad_is_running_ && !bprop_grad_stack_.empty() && !bprop_grad_stack_.top().second) {
    MS_LOG(DEBUG) << "Running custom bprop, no need to do GradPynativeOp.";
  } else {
    (void)ad::GradPynativeOp(top_cell()->k_pynative_cell_ptr(), cnode, input_args,
                             std::make_shared<ValueTuple>(input_args));
  }
}

AnfNodePtr GradExecutor::CreateMakeTupleGradNode(const ValuePtr &v, const std::string &obj_id) const {
  const auto &fg = top_cell()->fg();
  MS_EXCEPTION_IF_NULL(fg);
  MS_EXCEPTION_IF_NULL(v);
  ValuePtrList input_args;
  std::vector<AnfNodePtr> inputs{NewValueNode(prim::kPrimMakeTuple)};
  if (!v->isa<ValueSequence>()) {
    MS_LOG(DEBUG) << "The input obj is not a tuple or list.";
    return nullptr;
  }
  const auto &obj_tuple = v->cast<ValueSequencePtr>();
  const auto &v_list = obj_tuple->value();
  for (size_t i = 0; i < obj_tuple->size(); ++i) {
    const auto &v_arg = v_list[i];
    // Graph have no define for grad
    if (v_arg->isa<FuncGraph>()) {
      continue;
    }
    (void)input_args.emplace_back(v_arg);
    (void)inputs.emplace_back(GetInput(v_arg));
    (void)CreateMakeTupleGradNode(v_arg, PyNativeAlgo::Common::GetIdByValue(v_arg));
  }
  // Create make tuple node and record to graph info map.
  auto cnode = fg->NewCNode(inputs);
  MS_LOG(DEBUG) << "Create make tuple node: " << cnode->DebugString();
  RecordGradNodeToGraphInfoMap(fg, cnode, obj_id, input_args);
  return cnode;
}

AnfNodePtr GradExecutor::CreateTupleGetItemNode(const std::string &obj_id) const {
  const auto &fg = top_cell()->fg();
  // obj_id is obtained by calling the 'PyParser::GetIdByPyObj()'
  const auto graph_info = top_cell()->graph_info_map().at(fg);
  MS_EXCEPTION_IF_NULL(graph_info);
  if (graph_info->node_map.find(obj_id) == graph_info->node_map.end()) {
    MS_LOG(DEBUG) << "Can not find CNode for obj id: " << obj_id;
    return nullptr;
  }
  const auto &out = graph_info->node_map.at(obj_id);
  MS_LOG(DEBUG) << "Output size: " << out.second.size();
  auto c_node = out.first->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(c_node);
  auto abs = c_node->abstract();
  // Create tuple get item node
  for (const auto &idx : out.second) {
    std::vector<AnfNodePtr> tuple_get_item_inputs{NewValueNode(prim::kPrimTupleGetItem), c_node, NewValueNode(idx)};
    c_node = fg->NewCNode(tuple_get_item_inputs);
    if (abs != nullptr && abs->isa<abstract::AbstractTuple>()) {
      auto abs_tuple = dyn_cast<abstract::AbstractTuple>(abs);
      MS_EXCEPTION_IF_NULL(abs_tuple);
      const auto &elements = abs_tuple->elements();
      if (static_cast<size_t>(idx) >= elements.size()) {
        MS_LOG(EXCEPTION) << "Index exceeds the size of elements. Index " << idx << ", element size "
                          << elements.size();
      }
      auto prim_abs = elements[static_cast<size_t>(idx)];
      MS_EXCEPTION_IF_NULL(prim_abs);
      MS_LOG(DEBUG) << "Set tuple getitem abs " << prim_abs->ToString();
      c_node->set_abstract(prim_abs);
    }
  }
  MS_LOG(DEBUG) << "Create tuple get item node: " << c_node->DebugString();
  return c_node;
}

TopCellInfoPtr GradExecutor::GetTopCell(const std::string &already_run_cell_id) {
  TopCellInfoPtr find_top_cell = nullptr;
  // Complete match, means run grad operation first
  if (top_cell()->already_run_cell_id() == already_run_cell_id) {
    return top_cell();
  }
  // Partial match, means run forward first
  if (already_run_cell_id.find(top_cell()->already_run_cell_id()) != std::string::npos &&
      top_cell()->already_run_cell_id().back() == '_') {
    find_top_cell = top_cell();
  }
  // Same topcell info, but grad operation is not the same, construct backward graph again
  if (find_top_cell != nullptr) {
    if (!find_top_cell->grad_operation().empty() && find_top_cell->grad_operation() != grad_operation_) {
      MS_LOG(DEBUG) << "Already exist grad operation " << find_top_cell->grad_operation() << " is different with new "
                    << grad_operation_;
      return nullptr;
    } else {
      return find_top_cell;
    }
  }
  return nullptr;
}

void GradExecutor::SetHookChanged(const py::object &cell) const {
  if (top_cell_ == nullptr) {
    return;
  }
  const auto &cell_id = PyNativeAlgo::PyParser::GetIdByPyObj(cell);
  if (top_cell_->cell_id().find(cell_id) != std::string::npos) {
    top_cell_->set_hook_changed(true);
  }
  if (grad_flag_) {
    top_cell_->set_sub_cell_hook_changed(cell_id);
  }
}

void GradExecutor::ProcessOpGradInfo(const FrontendOpRunInfoPtr &op_run_info) const {
  MS_EXCEPTION_IF_NULL(op_run_info);
  if (!grad_flag_) {
    MS_LOG(DEBUG) << "Grad flag is false";
    return;
  }

  // Const value no need do op grad
  if (op_run_info->output_get_by_infer_value) {
    return;
  }
  // Do op grad and save node info
  if (custom_bprop_cell_count_ <= 0) {
    const auto &cnode = ConstructForwardGraph(op_run_info);
    MS_EXCEPTION_IF_NULL(cnode);
    cnode->set_abstract(op_run_info->base_op_run_info.abstract);
    SaveOutputNodeMap(op_run_info->out_value_id, op_run_info, cnode);
    DoOpGrad(op_run_info, cnode, op_run_info->out_value);
  }
}

void GradExecutor::SaveOutputNodeMap(const std::string &obj_id, const FrontendOpRunInfoPtr &op_run_info,
                                     const CNodePtr &cnode) const {
  MS_EXCEPTION_IF_NULL(op_run_info);
  if (input_args_info_stack_.empty()) {
    MS_LOG(DEBUG) << "No need save output";
    return;
  }
  MS_EXCEPTION_IF_NULL(cnode);
  MS_LOG(DEBUG) << "Cnode is " << cnode->DebugString() << ", out value id " << obj_id;
  // In hook compute, output is a copy of input; If hook input is a input param, follow op use hook output as input,
  // which GetInput will always find input param, so need delete from input param map
  if (kHookOp.find(op_run_info->base_op_run_info.op_name) != kHookOp.end()) {
    for (size_t i = 0; i < op_run_info->input_size; ++i) {
      top_cell()->DeleteParamNodeInfo(curr_g(), op_run_info->input_value_id[i]);
    }
  }
  top_cell()->SetNodeMapInGraphInfoMap(curr_g(), obj_id, cnode);
}

// Run ad grad for curr op and connect grad graph with previous op
void GradExecutor::DoOpGrad(const FrontendOpRunInfoPtr &op_run_info, const CNodePtr &cnode,
                            const ValuePtr &op_out) const {
  if (grad_is_running_ && !bprop_grad_stack_.top().second) {
    MS_LOG(DEBUG) << "Custom bprop, no need do op grad";
    return;
  }
  MS_EXCEPTION_IF_NULL(op_run_info);
  if (!ad::GradPynativeOp(top_cell()->k_pynative_cell_ptr(), cnode, op_run_info->input_value, op_out)) {
    MS_LOG(EXCEPTION) << "Failed to run ad grad for op " << op_run_info->base_op_run_info.op_name;
  }
}

AnfNodePtr GradExecutor::GetRealInputNodeBySkipHook(const AnfNodePtr &input_node) const {
  if (input_node == nullptr) {
    MS_LOG(DEBUG) << "The input node is nullptr.";
    return input_node;
  }
  const auto &cell_backward_hook_op = top_cell()->cell_backward_hook_op();
  for (const auto &elem : cell_backward_hook_op) {
    constexpr size_t cell_backward_hook_num = 2;
    if (elem.second.size() < cell_backward_hook_num) {  // In cell own scope, no need to skip backward hook op.
      continue;
    }
    // The input node is the first backward hook op of another cell, skip the backward hook op.
    if (IsPrimitiveCNode(input_node, prim::kPrimCellBackwardHook) && input_node == elem.second[0]) {
      // Single input.
      auto backward_hook_op = input_node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(backward_hook_op);
      return backward_hook_op->input(1);
    } else if (IsPrimitiveCNode(input_node, prim::kPrimTupleGetItem)) {
      // Multi inputs.
      auto tuple_get_item = input_node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(tuple_get_item);
      auto inp_in_tuple = tuple_get_item->input(1);
      MS_EXCEPTION_IF_NULL(inp_in_tuple);
      if (IsPrimitiveCNode(inp_in_tuple, prim::kPrimCellBackwardHook) && inp_in_tuple == elem.second[0]) {
        constexpr size_t idx = 2;
        auto idx_node = tuple_get_item->input(idx);
        MS_EXCEPTION_IF_NULL(idx_node);
        auto value_node = idx_node->cast<ValueNodePtr>();
        MS_EXCEPTION_IF_NULL(value_node);
        auto out_idx = GetValue<int64_t>(value_node->value());
        auto backward_hook_op = inp_in_tuple->cast<CNodePtr>();
        MS_EXCEPTION_IF_NULL(backward_hook_op);
        return backward_hook_op->input(1 + LongToSize(out_idx));
      }
    }
  }
  return input_node;
}

CNodePtr GradExecutor::ConstructForwardGraph(const FrontendOpRunInfoPtr &op_run_info) const {
  MS_EXCEPTION_IF_NULL(op_run_info);
  std::vector<AnfNodePtr> inputs;
  (void)inputs.emplace_back(NewValueNode(op_run_info->op_prim));
  for (size_t i = 0; i < op_run_info->input_size; i++) {
    AnfNodePtr input_node = nullptr;
    const auto node = GetInput(op_run_info->input_value[i]);
    input_node = GetRealInputNodeBySkipHook(node);
    // update abstract
    if (input_node != nullptr) {
      (void)inputs.emplace_back(input_node);
    }
  }
  const auto &cnode = top_cell()->fg()->NewCNodeInOrder(inputs);
  if (IsPrimitiveCNode(cnode, prim::kPrimCellBackwardHook)) {
    top_cell()->RecordCellBackwardHookOp(GetCurCellOrder(), cnode);
  }
  MS_LOG(DEBUG) << "Make CNode for " << op_run_info->base_op_run_info.op_name << ", new cnode is "
                << cnode->DebugString();
  return cnode;
}

void GradExecutor::SetBpropGraphJitLevel(const py::object &obj) const {
  if (!py::hasattr(obj, kAttrCellJitConfigDict)) {
    return;
  }

  auto jit_config = py::getattr(obj, kAttrCellJitConfigDict);
  if (!py::isinstance<py::dict>(jit_config)) {
    MS_LOG(EXCEPTION) << "JitConfig only support dict!";
  }
  auto jit_config_dict = jit_config.cast<py::dict>();
  auto graph_executor = pipeline::GraphExecutorPy::GetInstance();
  MS_EXCEPTION_IF_NULL(graph_executor);
  graph_executor->SetJitConfig(jit_config_dict);
}
}  // namespace pynative
}  // namespace mindspore
