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
#include "pipeline/pynative/pynative_utils.h"
#include <utility>
#include <vector>
#include "backend/common/optimizer/helper.h"
#include "backend/common/optimizer/op_adaptation_info_factory.h"
#include "pybind_api/ir/primitive_py.h"
#include "utils/ms_context.h"
#include "ir/cell.h"
#include "include/common/utils/utils.h"
#include "include/common/utils/convert_utils_py.h"
#include "include/common/debug/anf_ir_dump.h"
#include "pipeline/jit/parse/data_converter.h"

namespace mindspore {
namespace pynative {
namespace PyNativeAlgo {
std::string Common::GetIdByValue(const ValuePtr &v) {
  MS_EXCEPTION_IF_NULL(v);
  if (v->isa<tensor::Tensor>()) {
    return v->cast<tensor::TensorPtr>()->id();
  } else if (v->isa<Cell>()) {
    return v->cast<CellPtr>()->id();
  } else if (v->isa<mindspore::Type>()) {
    auto type_ptr = v->cast<mindspore::TypePtr>();
    return "Type:" + type_ptr->ToString();
  } else if (v->isa<StringImm>()) {
    return "S" + v->cast<StringImmPtr>()->value();
  } else if (v->isa<BoolImm>()) {
    return "B" + std::to_string(v->cast<BoolImmPtr>()->value());
  } else if (v->isa<IntegerImm>()) {
    return "I" + std::to_string(v->cast<Int64ImmPtr>()->value());
  } else if (v->isa<FloatImm>()) {
    return "F" + std::to_string(v->cast<FP32ImmPtr>()->value());
  } else if (v->isa<None>()) {
    return "None";
  } else if (v->isa<Ellipsis>()) {
    return "Ellipsis";
  } else if (v->isa<ValueSequence>()) {
    auto p_list = v->cast<ValueSequencePtr>();
    string prefix = v->isa<ValueTuple>() ? "Tuple<" : "List<";
    if (p_list->size() == 0) {
      prefix = "Empty:";
    } else {
      for (size_t i = 0; i < p_list->size(); ++i) {
        prefix += GetIdByValue(p_list->value()[i]) + ":";
      }
    }
    prefix.pop_back();
    prefix += ">";
    return prefix;
  }
  MS_LOG(DEBUG) << "Get type " << v->ToString();
  return v->ToString();
}

bool Common::ValueHasDynamicShape(const ValuePtr &value) {
  MS_EXCEPTION_IF_NULL(value);
  if (value->isa<tensor::Tensor>()) {
    return value->cast<tensor::TensorPtr>()->base_shape_ptr() != nullptr;
  } else if (value->isa<ValueSequence>()) {
    auto value_seq = value->cast<ValueSequencePtr>();
    return std::any_of(value_seq->value().begin(), value_seq->value().end(),
                       [](const ValuePtr &elem) { return ValueHasDynamicShape(elem); });
  }
  return false;
}

bool Common::IsTensor(const ValuePtr &v) {
  MS_EXCEPTION_IF_NULL(v);
  return v->isa<tensor::Tensor>() || v->isa<tensor::MetaSparseTensor>();
}

std::shared_ptr<PyNativeExecutor> Common::GetPyNativeExecutor() {
  const auto &executor = PyNativeExecutor::GetInstance();
  MS_EXCEPTION_IF_NULL(executor);
  return executor;
}

void Common::DumpGraphIR(const std::string &filename, const FuncGraphPtr &graph) {
#ifdef ENABLE_DUMP_IR
  const auto &ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (ms_context->get_param<bool>(MS_CTX_SAVE_GRAPHS_FLAG)) {
    DumpIR(filename, graph);
  }
#endif
}

std::string PyParser::GetPyObjId(const py::handle &obj) {
  py::object out = python_adapter::CallPyFn(parse::PYTHON_MOD_PARSE_MODULE, parse::PYTHON_MOD_GET_OBJ_ID, obj);
  if (py::isinstance<py::none>(out)) {
    MS_LOG(EXCEPTION) << "Get pyobj failed";
  }
  return out.cast<std::string>();
}

std::string PyParser::GetIdByPyObj(const py::object &obj) {
  if (py::isinstance<tensor::Tensor>(obj)) {
    return obj.cast<tensor::TensorPtr>()->id();
  } else if (py::isinstance<Cell>(obj)) {
    return obj.cast<CellPtr>()->id();
  } else if (py::isinstance<mindspore::Type>(obj)) {
    auto type_ptr = obj.cast<mindspore::TypePtr>();
    return "Type:" + type_ptr->ToString();
  } else if (py::isinstance<py::str>(obj)) {
    return "S" + obj.cast<std::string>();
  } else if (py::isinstance<py::bool_>(obj)) {
    return "B" + py::str(obj).cast<std::string>();
  } else if (py::isinstance<py::int_>(obj)) {
    return "I" + py::str(obj).cast<std::string>();
  } else if (py::isinstance<py::float_>(obj)) {
    return "F" + py::str(obj).cast<std::string>();
  } else if (py::isinstance<py::none>(obj)) {
    return "None";
  } else if (py::isinstance<py::ellipsis>(obj)) {
    return "Ellipsis";
  } else if (py::isinstance<py::tuple>(obj) || py::isinstance<py::list>(obj)) {
    auto p_list = py::cast<py::tuple>(obj);
    string prefix = py::isinstance<py::tuple>(obj) ? "Tuple<" : "List<";
    if (p_list.empty()) {
      prefix = "Empty:";
    } else {
      for (size_t i = 0; i < p_list.size(); ++i) {
        prefix += PyParser::GetIdByPyObj(p_list[i]) + ":";
      }
    }
    prefix.pop_back();
    prefix += ">";
    return prefix;
  }
  // For id with value and obj can be the same
  if (py::isinstance<tensor::CSRTensor>(obj) || py::isinstance<tensor::COOTensor>(obj) ||
      py::isinstance<tensor::RowTensor>(obj)) {
    return DataConvert::PyObjToValue(obj)->ToString();
  }
  return GetPyObjId(obj);
}

void PyParser::SetPrim(const FrontendOpRunInfoPtr &op_run_info, const py::object &prim_arg) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  const auto &adapter = prim_arg.cast<PrimitivePyAdapterPtr>();
  MS_EXCEPTION_IF_NULL(adapter);
  auto prim = adapter->attached_primitive();
  if (prim == nullptr) {
    prim = std::make_shared<PrimitivePy>(prim_arg, adapter);
    adapter->set_attached_primitive(prim);
  }
  if (!prim->HasPyObj()) {
    MS_LOG(EXCEPTION) << "Pyobj is empty";
  }
  op_run_info->op_prim = prim;
}

void PyParser::ParseOpInputByPythonObj(const FrontendOpRunInfoPtr &op_run_info, const py::list &op_inputs) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  op_run_info->input_size = op_inputs.size();
  op_run_info->input_value.resize(op_run_info->input_size);
  for (size_t i = 0; i < op_run_info->input_size; ++i) {
    op_run_info->input_value[i] = PyNativeAlgo::DataConvert::PyObjToValue(op_inputs[i]);
  }
}

std::string PyParser::GetCellId(const py::object &obj, const py::args &args, const InputArgsInfoPtr &input_args_info) {
  auto cell_id = GetIdByPyObj(obj);
  auto fn = [&cell_id](const abstract::AbstractBasePtr &abs) {
    MS_EXCEPTION_IF_NULL(abs);
    auto shape = abs->BuildShape();
    auto type = abs->BuildType();
    cell_id += "_" + shape->ToString();
    cell_id += type->ToString();
  };

  const auto &forward = Common::GetPyNativeExecutor()->forward_executor();
  const auto &node_abs_map = forward->NodeAbsMap();
  bool id_not_exist = (input_args_info == nullptr);
  for (size_t i = 0; i < args.size(); i++) {
    const auto &arg_id =
      id_not_exist ? PyNativeAlgo::PyParser::GetIdByPyObj(args[i]) : input_args_info->input_arg_id_vec[i];
    // Find in step process
    const auto it = node_abs_map.find(arg_id);
    if (it != node_abs_map.end()) {
      fn(it->second);
    } else {
      const auto &value = PyNativeAlgo::DataConvert::PyObjToValue(args[i]);
      auto abs = value->ToAbstract();
      MS_EXCEPTION_IF_NULL(abs);
      if (abs->isa<abstract::AbstractTensor>()) {
        abs->set_value(kAnyValue);
      }
      forward->SetNodeAbsMapById(arg_id, abs);
      fn(abs);
    }
  }
  return cell_id;
}

InputArgsInfoPtr PyParser::GetInputArgsInfo(const py::object &obj, const py::args &args, bool is_top_grad_cell) {
  bool has_custom_bprop = py::hasattr(obj, parse::CUSTOM_BPROP_NAME);
  const auto &obj_id = PyNativeAlgo::PyParser::GetIdByPyObj(obj);
  const auto &input_args_info = std::make_shared<InputArgsInfo>(has_custom_bprop, args.size(), obj_id);
  for (size_t i = 0; i < args.size(); i++) {
    const auto &id = PyNativeAlgo::PyParser::GetIdByPyObj(args[i]);
    (void)input_args_info->input_arg_id_vec.emplace_back(id);
  }
  // Only the top cell requires value conversion
  if (is_top_grad_cell) {
    for (size_t i = 0; i < args.size(); ++i) {
      input_args_info->input_args_id += input_args_info->input_arg_id_vec[i] + "_";
      const auto &v = PyNativeAlgo::DataConvert::PyObjToValue(args[i]);
      (void)input_args_info->input_arg_value_vec.emplace_back(v);
    }
  }
  input_args_info->cell_id = GetCellId(obj, args, input_args_info);
  MS_LOG(DEBUG) << "cell_id is " << obj_id << ", is top grad cell " << is_top_grad_cell;
  return input_args_info;
}

py::object DataConvert::ValueToPyObj(const ValuePtr &v) { return ValueToPyData(v); }

ValuePtr DataConvert::PyObjToValue(const py::object &obj) {
  ValuePtr converted_ret = parse::data_converter::PyDataToValue(obj);
  if (!converted_ret) {
    MS_LOG(EXCEPTION) << "Attribute convert error with type: " << std::string(py::str(obj));
  }
  return converted_ret;
}

ValuePtr DataConvert::BaseRefToValue(const BaseRef &value) {
  MS_EXCEPTION_IF_NULL(value);
  ValuePtr ret;
  if (utils::isa<tensor::TensorPtr>(value)) {
    ret = utils::cast<tensor::TensorPtr>(value);
  } else if (utils::isa<ValuePtr>(value)) {
    ret = utils::cast<ValuePtr>(value);
  } else if (utils::isa<VectorRef>(value)) {
    auto vec_ref = utils::cast<VectorRef>(value);
    ret = VectorRefToValue(vec_ref);
  } else if (utils::isa<int>(value)) {
    ret = MakeValue(utils::cast<int>(value));
  } else if (utils::isa<float>(value)) {
    ret = MakeValue(utils::cast<float>(value));
  } else if (utils::isa<double>(value)) {
    ret = MakeValue(utils::cast<double>(value));
  } else if (utils::isa<bool>(value)) {
    ret = MakeValue(utils::cast<bool>(value));
  } else {
    MS_LOG(EXCEPTION) << "value is not support type " << value.ToString();
  }
  return ret;
}

ValuePtr DataConvert::VectorRefToValue(const VectorRef &vec_ref) {
  MS_EXCEPTION_IF_NULL(vec_ref);
  size_t value_size = vec_ref.size();
  std::vector<ValuePtr> v_list(value_size);
  for (size_t i = 0; i < value_size; ++i) {
    v_list[i] = BaseRefToValue(vec_ref[i]);
  }
  return std::make_shared<ValueTuple>(v_list);
}

void DataConvert::FlattenTupleArg(const ValuePtr &v, std::vector<ValuePtr> *flatten_v) {
  MS_EXCEPTION_IF_NULL(v);
  MS_EXCEPTION_IF_NULL(flatten_v);
  const auto &v_vec = v->cast<ValueSequencePtr>();
  size_t v_vec_size = v_vec->size();
  for (size_t i = 0; i < v_vec_size; ++i) {
    const auto &elem_v = v_vec->value()[i];
    if (elem_v->isa<ValueSequence>()) {
      FlattenTupleArg(elem_v, flatten_v);
    } else if (PyNativeAlgo::Common::IsTensor(elem_v)) {
      (void)flatten_v->emplace_back(elem_v);
    }
  }
}

void DataConvert::FlattenArgs(const std::vector<ValuePtr> &v_vec, std::vector<ValuePtr> *flatten_v) {
  MS_EXCEPTION_IF_NULL(flatten_v);
  for (size_t i = 0; i < v_vec.size(); ++i) {
    const auto &v = v_vec[i];
    MS_EXCEPTION_IF_NULL(v);
    if (v->isa<ValueSequence>()) {
      FlattenTupleArg(v, flatten_v);
    } else if (PyNativeAlgo::Common::IsTensor(v)) {
      (void)flatten_v->emplace_back(v);
    }
  }
}

bool DataConvert::RunOpConvertConstInputToAttr(const FrontendOpRunInfoPtr &op_run_info, const ValuePtr &v,
                                               size_t input_index, const PrimitivePtr &op_prim,
                                               const mindspore::HashSet<size_t> &input_attrs) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(v);
  MS_EXCEPTION_IF_NULL(op_prim);
  if (input_attrs.find(input_index) == input_attrs.end()) {
    return false;
  }
  const auto &input_names_value = op_prim->GetAttr(kAttrInputNames);
  if (input_names_value == nullptr) {
    return false;
  }
  const auto &input_names_vec = GetValue<std::vector<std::string>>(input_names_value);
  if (input_index >= input_names_vec.size()) {
    MS_LOG(EXCEPTION) << "The input index: " << input_index << " is larger than the input names vector size!";
  }
  const auto &input_name = input_names_vec[input_index];
  if (v->isa<tensor::Tensor>()) {
    auto tensor = v->cast<tensor::TensorPtr>();
    if (tensor->data().const_data() == nullptr) {
      return false;
    }
  }
  (void)op_prim->AddAttr(input_name, v);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto backend = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  if (op_prim->name() == "UnsortedSegmentSum" && backend == kAscendDevice) {
    op_prim->set_name("UnsortedSegmentSumD");
  }
  return true;
}

void DataConvert::PlantTensorTupleToVector(const FrontendOpRunInfoPtr &op_run_info, const ValueSequencePtr &value_seq,
                                           const PrimitivePtr &op_prim, size_t index) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(value_seq);
  MS_EXCEPTION_IF_NULL(op_prim);
  for (const auto &v : value_seq->value()) {
    if (!v->isa<tensor::Tensor>()) {
      MS_LOG(EXCEPTION) << "The input object is not a tensor!";
    }
    int64_t tensor_mask = kParameterDataTensorMask;
    auto tensor = v->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(tensor);
    if (tensor->is_parameter()) {
      tensor_mask = kParameterWeightTensorMask;
    }
    (void)op_run_info->base_op_run_info.input_tensor.emplace_back(tensor);
    (void)op_run_info->base_op_run_info.input_mask.emplace_back(tensor_mask);
  }
  if (op_prim->HasAttr(kAttrDynInputSizes)) {
    int64_t elem_size = SizeToLong(value_seq->size());
    auto dyn_v = GetValue<const std::vector<int64_t>>(op_prim->GetAttr(kAttrDynInputSizes));
    if (dyn_v.size() != op_run_info->input_size) {
      for (size_t i = dyn_v.size(); i < index; ++i) {
        (void)dyn_v.emplace_back(-1);
      }
      (void)dyn_v.emplace_back(elem_size);
      (void)op_prim->set_attr(kAttrDynInputSizes, MakeValue(dyn_v));
    } else {
      if (dyn_v[index] != elem_size) {
        dyn_v[index] = elem_size;
        op_prim->set_attr(kAttrDynInputSizes, MakeValue(dyn_v));
      }
    }
  } else {
    std::vector<int64_t> dyn_v;
    for (size_t i = 0; i < index; ++i) {
      (void)dyn_v.emplace_back(-1);
    }
    (void)dyn_v.emplace_back(SizeToLong(value_seq->size()));
    op_prim->set_attr(kAttrDynInputSizes, MakeValue(dyn_v));
  }
}

void DataConvert::ConvertValueTupleToTensor(const FrontendOpRunInfoPtr &op_run_info,
                                            const ValueSequencePtr &value_seq) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(value_seq);
  ValueTuplePtr value_tuple;
  if (value_seq->isa<ValueList>()) {
    value_tuple = std::make_shared<ValueTuple>(value_seq->value());
  } else {
    value_tuple = value_seq->cast<ValueTuplePtr>();
  }
  MS_EXCEPTION_IF_NULL(value_tuple);
  auto tensor_ptr = opt::CreateTupleTensor(value_tuple);
  MS_EXCEPTION_IF_NULL(tensor_ptr);
  (void)op_run_info->base_op_run_info.input_tensor.emplace_back(tensor_ptr);
}

void DataConvert::ConvertCSRTensorToTensorList(const FrontendOpRunInfoPtr &op_run_info,
                                               const tensor::CSRTensorPtr &csr_tensor, const PrimitivePtr &op_prim) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(op_prim);
  MS_EXCEPTION_IF_NULL(csr_tensor);
  constexpr int input_num = 3;
  const auto input_names = op_prim->GetAttr(kAttrInputNames);
  if (input_names == nullptr) {
    MS_LOG(DEBUG) << "input_names are nullptr";
    return;
  }
  (void)op_run_info->base_op_run_info.input_tensor.emplace_back(csr_tensor->GetIndptr());
  (void)op_run_info->base_op_run_info.input_tensor.emplace_back(csr_tensor->GetIndices());
  (void)op_run_info->base_op_run_info.input_tensor.emplace_back(csr_tensor->GetValues());
  const auto it = op_run_info->base_op_run_info.input_mask.end();
  (void)op_run_info->base_op_run_info.input_mask.insert(it, input_num, kParameterDataTensorMask);
  op_prim->set_attr("is_csr", MakeValue(true));
  op_prim->set_attr("dense_shape", MakeValue(csr_tensor->shape()));
}

void DataConvert::ConvertTupleValueToTensor(const FrontendOpRunInfoPtr &op_run_info, const ValueSequencePtr &value_seq,
                                            const PrimitivePtr &op_prim, size_t index) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(value_seq);
  MS_EXCEPTION_IF_NULL(op_prim);

  const auto &tuple_inputs = value_seq->value();
  if (tuple_inputs.empty()) {
    std::vector<int64_t> axis = {};
    (void)op_run_info->base_op_run_info.input_tensor.emplace_back(std::make_shared<tensor::Tensor>(axis, kInt64));
    (void)op_run_info->base_op_run_info.input_mask.emplace_back(kValueNodeTensorMask);
    return;
  }
  if (tuple_inputs[0]->isa<tensor::Tensor>()) {
    PlantTensorTupleToVector(op_run_info, value_seq, op_prim, index);
  } else {
    ConvertValueTupleToTensor(op_run_info, value_seq);
    (void)op_run_info->base_op_run_info.input_mask.emplace_back(kValueNodeTensorMask);
  }
}

void DataConvert::ConvertValueToTensor(const FrontendOpRunInfoPtr &op_run_info, const ValuePtr &v, size_t index,
                                       const PrimitivePtr &op_prim) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(v);
  MS_EXCEPTION_IF_NULL(op_prim);
  tensor::TensorPtr tensor_ptr = nullptr;
  int64_t tensor_mask = kParameterDataTensorMask;
  if (v->isa<tensor::Tensor>()) {
    tensor_ptr = v->cast<tensor::TensorPtr>();
    if (tensor_ptr->is_parameter()) {
      tensor_mask = kParameterWeightTensorMask;
    }
  } else if (v->isa<FloatImm>()) {
    double input_value = v->cast<FP32ImmPtr>()->value();
    tensor_ptr = std::make_shared<tensor::Tensor>(input_value, kFloat32);
    tensor_mask = kValueNodeTensorMask;
  } else if (v->isa<BoolImm>()) {
    tensor_ptr = std::make_shared<tensor::Tensor>(v->cast<BoolImmPtr>()->value(), kBool);
    tensor_mask = kValueNodeTensorMask;
  } else if (v->isa<IntegerImm>()) {
    int64_t input = v->cast<Int64ImmPtr>()->value();
    if (op_prim->name() == prim::kPrimCSRReduceSum->name()) {
      op_prim->set_attr("axis", MakeValue(input));
      return;
    }
    tensor_ptr = std::make_shared<tensor::Tensor>(input, kInt64);
    tensor_mask = kValueNodeTensorMask;
  } else if (v->isa<Type>()) {
    int64_t type_id = v->cast<TypePtr>()->type_id();
    tensor_ptr = std::make_shared<tensor::Tensor>(type_id, kInt64);
    tensor_ptr->set_user_data(kTensorValueIsType, v);
    tensor_mask = kValueNodeTensorMask;
  } else if (v->isa<StringImm>()) {
    auto value_string = GetValue<std::string>(v);
    const ShapeVector shape = {1, SizeToLong(value_string.size())};
    tensor_ptr = std::make_shared<tensor::Tensor>(kObjectTypeString, shape, value_string.data(), value_string.size());
    tensor_mask = kValueNodeTensorMask;
  } else if (v->isa<ValueSequence>()) {
    ConvertTupleValueToTensor(op_run_info, v->cast<ValueSequencePtr>(), op_prim, index);
    return;
  } else if (v->isa<tensor::CSRTensor>()) {
    ConvertCSRTensorToTensorList(op_run_info, v->cast<tensor::CSRTensorPtr>(), op_prim);
    return;
  } else if (v->isa<None>() || v->isa<Monad>()) {
    return;
  } else {
    MS_LOG(EXCEPTION) << "Run op inputs type is invalid!";
  }
  MS_EXCEPTION_IF_NULL(tensor_ptr);
  (void)op_run_info->base_op_run_info.input_tensor.emplace_back(tensor_ptr);
  (void)op_run_info->base_op_run_info.input_mask.emplace_back(tensor_mask);
}

bool DataConvert::NeedConvertConstInputToAttr(const FrontendOpRunInfoPtr &op_run_info, const std::string &device_target,
                                              mindspore::HashSet<size_t> *input_to_attr_ptr) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(input_to_attr_ptr);
  if (op_run_info->base_op_run_info.op_name == prim::kPrimCustom->name()) {
    // Custom op needs to set reg dynamically
    mindspore::HashSet<size_t> attr_indexes;
    const PrimitivePtr &op_prim = op_run_info->op_prim;
    MS_EXCEPTION_IF_NULL(op_prim);
    opt::GetCustomOpAttrIndex(op_prim, input_to_attr_ptr);
    return !input_to_attr_ptr->empty();
  }

  // Ascend const input to attr move to AscendVmOpAdapter
  if (device_target == kAscendDevice) {
    return false;
  }

  auto reg_info = opt::OpAdaptationInfoRegister::GetInstance().GetOpAdaptationInfo(
    op_run_info->base_op_run_info.op_name, device_target, true);
  if (reg_info == nullptr) {
    return false;
  } else {
    for (auto &iter : reg_info->GetInputAttrInfoMap()) {
      (void)input_to_attr_ptr->insert(iter.second.GetInputIndex());
    }
  }
  return !input_to_attr_ptr->empty();
}

void DataConvert::GetInputTensor(const FrontendOpRunInfoPtr &op_run_info, const std::string &device_target) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  mindspore::HashSet<size_t> input_to_attr = {};
  bool need_convert_input_to_attr = NeedConvertConstInputToAttr(op_run_info, device_target, &input_to_attr);
  MS_LOG(DEBUG) << "Need convert input to addr " << need_convert_input_to_attr;
  if (need_convert_input_to_attr) {
    // Clone a new prim
    op_run_info->op_prim = std::make_shared<PrimitivePy>(*(op_run_info->op_prim));
    MS_EXCEPTION_IF_NULL(op_run_info->op_prim->adapter());
    if (op_run_info->op_prim->adapter()->attached_primitive() == nullptr) {
      op_run_info->op_prim->adapter()->set_attached_primitive(op_run_info->op_prim);
    }
  }
  const auto &op_prim = op_run_info->op_prim;

  // Get input tensors.
  op_prim->BeginRecordAddAttr();
  for (size_t index = 0; index < op_run_info->input_size; ++index) {
    const ValuePtr &input_object = op_run_info->input_value[index];
    // convert const input to attr
    if (need_convert_input_to_attr &&
        RunOpConvertConstInputToAttr(op_run_info, input_object, index, op_prim, input_to_attr)) {
      continue;
    }
    // Mark tensors, common tensor data : 0, weight param: 1, valuenode(float_, int_): 2
    ConvertValueToTensor(op_run_info, input_object, index, op_prim);
    // -1 indicates input_object is not a dynInput
    if (op_prim->HasAttr(kAttrDynInputSizes) && !input_object->isa<ValueSequence>()) {
      auto dyn_v = GetValue<const std::vector<int64_t>>(op_prim->GetAttr(kAttrDynInputSizes));
      (void)dyn_v.emplace_back(-1);
      op_prim->set_attr(kAttrDynInputSizes, MakeValue(dyn_v));
    }
  }
  op_prim->EndRecordAddAttr();
}
}  // namespace PyNativeAlgo
}  // namespace pynative
}  // namespace mindspore
