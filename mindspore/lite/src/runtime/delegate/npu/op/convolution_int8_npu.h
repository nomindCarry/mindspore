/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_NPU_OP_CONVOLUTION_INT8_NPU_H_
#define MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_NPU_OP_CONVOLUTION_INT8_NPU_H_

#include <vector>
#include <string>
#include <utility>
#include <unordered_map>
#include "include/graph/op/all_ops.h"
#include "src/runtime/delegate/npu/op/convolution_base_npu.h"
namespace mindspore::lite {
class ConvolutionInt8NPUOp : public ConvolutionBaseNPUOp {
 public:
  ConvolutionInt8NPUOp(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                       const std::vector<mindspore::MSTensor> &out_tensors, const std::string &name)
      : ConvolutionBaseNPUOp(primitive, in_tensors, out_tensors, name) {}

  ~ConvolutionInt8NPUOp() override;

  int IsSupport(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                const std::vector<mindspore::MSTensor> &out_tensors) override;

  int Init(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
           const std::vector<mindspore::MSTensor> &out_tensors) override;

  int SetNPUInputs(const std::vector<mindspore::MSTensor> &in_tensors,
                   const std::vector<mindspore::MSTensor> &out_tensors, const std::vector<ge::Operator *> &npu_inputs,
                   const std::unordered_map<int, std::pair<ge::Operator *, int>> &index2_multi_out_index) override;

  ge::Operator *GetNPUOp() override;

 private:
  int SetConvParam(const schema::Conv2DFusion *conv_prim);

  schema::ActivationType act_type_ = schema::ActivationType_NO_ACTIVATION;
  hiai::op::QuantizedConvolution *conv_ = nullptr;
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_NPU_OP_CONVOLUTION_NPU_H_
