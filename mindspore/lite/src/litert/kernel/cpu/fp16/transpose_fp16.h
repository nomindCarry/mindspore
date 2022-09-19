/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP16_TRANSPOSE_FP16_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP16_TRANSPOSE_FP16_H_

#include <arm_neon.h>
#include <vector>
#include "src/litert/kernel/cpu/base/transpose_base.h"

namespace mindspore::kernel {

class TransposeFp16CPUKernel : public TransposeBaseCPUKernel {
 public:
  explicit TransposeFp16CPUKernel(OpParameter *param, const std::vector<lite::Tensor *> &inputs,
                                  const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : TransposeBaseCPUKernel(param, inputs, outputs, ctx) {}
  ~TransposeFp16CPUKernel() = default;

  int ReSize() override;
  int DoTransposeMultiThread(int task_id) override;

 private:
  int DoTransposeSingleThread() override;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP16_TRANSPOSE_FP16_H_
