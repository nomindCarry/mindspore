/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#include "pipeline/pynative/pynative_execute.h"
#include "pipeline/pynative/pynative_utils.h"
#include "pipeline/jit/debug/trace.h"
#include "pybind_api/pybind_patch.h"
#include "include/common/utils/config_manager.h"
#include "include/common/pybind_api/api_register.h"
#include "frontend/optimizer/ad/grad.h"
#include "pipeline/jit/pass.h"
#include "runtime/pynative/op_executor.h"
#include "runtime/pynative/op_compiler.h"
#include "ir/cell.h"

namespace mindspore::pynative {
std::shared_ptr<PyNativeExecutor> PyNativeExecutor::executor_ = nullptr;
ForwardExecutorPtr PyNativeExecutor::forward_executor_ = nullptr;
GradExecutorPtr PyNativeExecutor::grad_executor_ = nullptr;
std::mutex PyNativeExecutor::instance_lock_;

namespace {
template <typename T, typename... Args>
T PyNativeExecutorTry(const std::function<T(const Args &...)> &method, const Args &...args) {
  const auto &inst = PyNativeExecutor::GetInstance();
  MS_EXCEPTION_IF_NULL(inst);
  MS_EXCEPTION_IF_NULL(method);
  try {
    return method(args...);
  } catch (const py::error_already_set &ex) {
    // print function call stack info before release
    std::ostringstream oss;
    trace::TraceGraphEval();
    trace::GetEvalStackInfo(oss);
    // call py::print to output function call stack to STDOUT, in case of output the log to file, the user can see
    // these info from screen, no need to open log file to find these info
    py::print(oss.str());
    MS_LOG(ERROR) << oss.str();
    inst->ClearRes();
    // re-throw this exception to Python interpreter to handle it
    throw(py::error_already_set(ex));
  } catch (const py::index_error &ex) {
    inst->ClearRes();
    throw py::index_error(ex);
  } catch (const py::value_error &ex) {
    inst->ClearRes();
    throw py::value_error(ex);
  } catch (const py::type_error &ex) {
    inst->ClearRes();
    throw py::type_error(ex);
  } catch (const py::name_error &ex) {
    inst->ClearRes();
    throw py::name_error(ex);
  } catch (const std::exception &ex) {
    inst->ClearRes();
    // re-throw this exception to Python interpreter to handle it
    throw(std::runtime_error(ex.what()));
  } catch (...) {
    inst->ClearRes();
#ifndef _MSC_VER
    auto exception_type = abi::__cxa_current_exception_type();
    MS_EXCEPTION_IF_NULL(exception_type);
    std::string ex_name(exception_type->name());
    MS_LOG(EXCEPTION) << "Error occurred when compile graph. Exception name: " << ex_name;
#else
    MS_LOG(EXCEPTION) << "Error occurred when compile graph.";
#endif
  }
}
}  // namespace

py::object PyNativeExecutor::RealRunOp(const py::args &args) const {
  FrontendOpRunInfoPtr op_run_info = forward_executor()->GenerateOpRunInfo(args);
  PyNativeExecutorTry(forward_executor()->RunOpS, op_run_info);
  if (PyGILState_Check() == 0) {
    py::gil_scoped_acquire acquire;
    return PyNativeAlgo::DataConvert::ValueToPyObj(op_run_info->out_value);
  } else {
    return PyNativeAlgo::DataConvert::ValueToPyObj(op_run_info->out_value);
  }
}

py::object PyNativeExecutor::CallConstantFolding(const py::args &args) const {
  return forward_executor()->infer_operation()->CallConstantFolding(args);
}

void PyNativeExecutor::set_py_exe_path(const py::object &py_exe_path) const {
  if (!py::isinstance<py::str>(py_exe_path)) {
    MS_LOG(EXCEPTION) << "Failed, py_exe_path input is not a str";
  }
  const auto &py_exe_path_s = py_exe_path.cast<std::string>();
  auto ms_context = MsContext::GetInstance();
  ms_context->set_param<std::string>(MS_CTX_PYTHON_EXE_PATH, py_exe_path_s);
}

void PyNativeExecutor::set_kernel_build_server_dir(const py::object &kernel_build_server_dir) const {
  if (!py::isinstance<py::str>(kernel_build_server_dir)) {
    MS_LOG(EXCEPTION) << "Failed, kernel_build_server_dir input is not a str";
  }
  const auto &kernel_build_server_dir_s = kernel_build_server_dir.cast<std::string>();
  auto ms_context = MsContext::GetInstance();
  ms_context->set_param<std::string>(MS_CTX_KERNEL_BUILD_SERVER_DIR, kernel_build_server_dir_s);
}

void PyNativeExecutor::ClearRes() const {
  runtime::OpExecutor::GetInstance().Reset();
  pynative::OpCompiler::GetInstance().ClearAllCache();

  // Maybe exit in runop step
  auto ms_context = MsContext::GetInstance();
  if (ms_context != nullptr) {
    ms_context->set_param<bool>(MS_CTX_ENABLE_PYNATIVE_INFER, false);
  }
  ConfigManager::GetInstance().ResetIterNum();
  if (forward_executor_ != nullptr) {
    forward_executor_->ClearRes();
  }
  if (grad_executor_ != nullptr) {
    grad_executor_->ClearRes();
  }
  ad::CleanRes();
  pipeline::ReclaimOptimizer();
  MS_LOG(DEBUG) << "Clear all res";
}

void PyNativeExecutor::Init() {
  MS_LOG(DEBUG) << "Init PyNativeExecutor";
  forward_executor_ = std::make_shared<ForwardExecutor>();
  grad_executor_ = std::make_shared<GradExecutor>(forward_executor_);
  forward_executor_->set_grad_executor(grad_executor_);
}

void PyNativeExecutor::Sync() const { forward_executor()->Sync(); }

void PyNativeExecutor::SetHookChanged(const py::object &cell) const {
  if (!py::isinstance<Cell>(cell)) {
    MS_LOG(EXCEPTION) << "The 'set_hook_changed' function is only supported on Cell object!";
  }
  grad_executor()->SetHookChanged(cell);
}

void PyNativeExecutor::set_graph_phase(const std::string &graph_phase) const {
  grad_executor()->ms_function()->set_graph_phase(graph_phase);
}

bool PyNativeExecutor::grad_flag() const { return grad_executor()->grad_flag(); }

void PyNativeExecutor::set_grad_flag(bool flag) const { grad_executor()->set_grad_flag(flag); }

py::object PyNativeExecutor::CheckAlreadyRun(const prim::GradOperationPtr &grad, const py::object &obj,
                                             const py::object &grad_hash_id, const py::args &args) const {
  return grad_executor()->CheckAlreadyRun(grad, obj, grad_hash_id, args);
}

void PyNativeExecutor::NewGraph(const py::object &obj, const py::args &args) const {
  forward_executor()->ProcessBeforeNewGraph(obj, args);

  if (!grad_flag()) {
    MS_LOG(DEBUG) << "Grad flag is false";
    return;
  }
  PyNativeExecutorTry(grad_executor()->InitGraph, obj, args);
}

void PyNativeExecutor::EndGraph(const py::object &obj, const py::object &out, const py::args &args) const {
  bool is_cell = py::isinstance<Cell>(obj);
  forward_executor()->ProcessBeforeEndGraph(obj, is_cell);

  if (!grad_flag()) {
    MS_LOG(DEBUG) << "Grad flag is false";
    return;
  }
  PyNativeExecutorTry(grad_executor()->LinkGraph, obj, out, args);
  forward_executor()->ProcessAfterEndGraph(obj, is_cell);
}

py::object PyNativeExecutor::Run() const { return PyNativeExecutorTry(grad_executor()->RunGraph); }

void PyNativeExecutor::GradNet(const prim::GradOperationPtr &grad, const py::object &cell, const py::object &weights,
                               const py::object &grad_position, const py::args &args) const {
  PyNativeExecutorTry(grad_executor()->GradGraph, grad, cell, weights, grad_position, args);
}

py::object PyNativeExecutor::GradMsFunction(const py::object &out, const py::args &args) const {
  return grad_executor()->ms_function()->GradMsFunction(out, args);
}

void PyNativeExecutor::SetLazyBuild(bool enable) const { forward_executor()->set_lazy_build(enable); }

bool PyNativeExecutor::IsFirstCell() const { return forward_executor()->IsFirstCell(); }

void RegPyNativeExecutor(const py::module *m) {
  (void)py::class_<PyNativeExecutor, std::shared_ptr<PyNativeExecutor>>(*m, "PyNativeExecutor_")
    .def_static("get_instance", &PyNativeExecutor::GetInstance, "PyNativeExecutor get_instance.")
    .def("is_first_cell", &PyNativeExecutor::IsFirstCell, "check if the first cell.")
    .def("new_graph", &PyNativeExecutor::NewGraph, "pynative new a graph.")
    .def("end_graph", &PyNativeExecutor::EndGraph, "pynative end a graph.")
    .def("check_run", &PyNativeExecutor::CheckAlreadyRun, "pynative check graph run before.")
    .def("grad_ms_function", &PyNativeExecutor::GradMsFunction, "pynative grad for ms_function.")
    .def("grad_net", &PyNativeExecutor::GradNet, "pynative grad graph.")
    .def("clear_res", &PyNativeExecutor::ClearRes, "pynative clear exception res.")
    .def("sync", &PyNativeExecutor::Sync, "pynative sync stream.")
    .def("set_lazy_build", &PyNativeExecutor::SetLazyBuild, "pynative build kernel async")
    .def("__call__", &PyNativeExecutor::Run, "pynative executor run grad graph.")
    .def("set_graph_phase", &PyNativeExecutor::set_graph_phase, "pynative set graph phase")
    .def("grad_flag", &PyNativeExecutor::grad_flag, "pynative grad flag")
    .def("set_hook_changed", &PyNativeExecutor::SetHookChanged, "set pynative hook changed")
    .def("set_grad_flag", &PyNativeExecutor::set_grad_flag, py::arg("flag") = py::bool_(false),
         "Executor set grad flag.")
    .def("set_py_exe_path", &PyNativeExecutor::set_py_exe_path, py::arg("py_exe_path") = py::str(""),
         "set python executable path.")
    .def("set_kernel_build_server_dir", &PyNativeExecutor::set_kernel_build_server_dir,
         py::arg("kernel_build_server_dir") = py::str(""), "set kernel build server directory path.")
    .def("real_run_op", &PyNativeExecutor::RealRunOp, "Run op pynatively.")
    .def("constant_folding", &PyNativeExecutor::CallConstantFolding, "Call Constant Folding Primitive");
}
}  // namespace mindspore::pynative
