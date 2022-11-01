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

#ifndef MINDSPORE_CCSRC_PIPELINE_PYNATIVE_PYNATIVE_UTILS_H_
#define MINDSPORE_CCSRC_PIPELINE_PYNATIVE_PYNATIVE_UTILS_H_

#include <memory>
#include <string>
#include <vector>
#include "pipeline/pynative/base.h"
#include "pipeline/pynative/pynative_execute.h"

namespace mindspore {
namespace pynative {
class PyNativeExecutor;

namespace PyNativeAlgo {
// Common function
struct Common {
  static std::string GetIdByValue(const ValuePtr &v);
  static bool ValueHasDynamicShape(const ValuePtr &value);
  static bool IsTensor(const ValuePtr &v);
  static void DumpGraphIR(const std::string &filename, const FuncGraphPtr &graph);
  static std::shared_ptr<PyNativeExecutor> GetPyNativeExecutor();
};

// Parser python
struct PyParser {
  static std::string GetPyObjId(const py::handle &obj);
  static std::string GetIdByPyObj(const py::object &obj);
  static void SetPrim(const FrontendOpRunInfoPtr &op_run_info, const py::object &prim_arg);
  static void ParseOpInputByPythonObj(const FrontendOpRunInfoPtr &op_run_info, const py::list &op_inputs);
  static std::string GetCellId(const py::object &obj, const py::args &args,
                               const InputArgsInfoPtr &input_args_info = nullptr);
  static InputArgsInfoPtr GetInputArgsInfo(const py::object &obj, const py::args &args, bool is_top_grad_cell);
};

// Data convert
struct DataConvert {
  static py::object ValueToPyObj(const ValuePtr &v);
  static ValuePtr PyObjToValue(const py::object &obj);
  static ValuePtr BaseRefToValue(const BaseRef &value);
  static ValuePtr VectorRefToValue(const VectorRef &vec_ref);
  static void FlattenTupleArg(const ValuePtr &v, std::vector<ValuePtr> *flatten_v);
  static void FlattenArgs(const std::vector<ValuePtr> &v_vec, std::vector<ValuePtr> *flatten_v);
  static void GetInputTensor(const FrontendOpRunInfoPtr &op_run_info, const std::string &device_target);
  static void ConvertCSRTensorToTensorList(const FrontendOpRunInfoPtr &op_run_info,
                                           const tensor::CSRTensorPtr &csr_tensor, const PrimitivePtr &op_prim);
  static void ConvertValueTupleToTensor(const FrontendOpRunInfoPtr &op_run_info, const ValueSequencePtr &value_seq);
  static void PlantTensorTupleToVector(const FrontendOpRunInfoPtr &op_run_info, const ValueSequencePtr &value_seq,
                                       const PrimitivePtr &op_prim, size_t index);
  static void ConvertTupleValueToTensor(const FrontendOpRunInfoPtr &op_run_info, const ValueSequencePtr &value_seq,
                                        const PrimitivePtr &op_prim, size_t index);
  static void ConvertValueToTensor(const FrontendOpRunInfoPtr &op_run_info, const ValuePtr &v, size_t index,
                                   const PrimitivePtr &op_prim);
  static bool NeedConvertConstInputToAttr(const FrontendOpRunInfoPtr &op_run_info, const std::string &device_target,
                                          mindspore::HashSet<size_t> *input_to_attr_ptr);
  static bool RunOpConvertConstInputToAttr(const FrontendOpRunInfoPtr &op_run_info, const ValuePtr &v,
                                           size_t input_index, const PrimitivePtr &op_prim,
                                           const mindspore::HashSet<size_t> &input_attrs);
};
};  // namespace PyNativeAlgo
}  // namespace pynative
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PIPELINE_PYNATIVE_PYNATIVE_UTILS_H_
