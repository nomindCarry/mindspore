/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_SRC_LITERT_LITE_KERNEL_MOD_H_
#define MINDSPORE_LITE_SRC_LITERT_LITE_KERNEL_MOD_H_

#include <memory>
#include <vector>
#include <string>
#include "src/litert/lite_kernel.h"
#include "src/litert/kernel_exec.h"
#include "kernel/kernel.h"
#include "include/model.h"

namespace mindspore::kernel {
class LiteKernelMod : public LiteKernel {
 public:
  explicit LiteKernelMod(std::shared_ptr<mindspore::kernel::KernelMod> kernel_mod, const CNodePtr &cnode,
                         const kernel::BaseOperatorPtr &base_operator, std::vector<lite::Tensor *> in_tensors,
                         std::vector<lite::Tensor *> out_tensors, const lite::InnerContext *ctx)
      : LiteKernel(nullptr, in_tensors, out_tensors, ctx),
        kernel_mod_(kernel_mod),
        cnode_(cnode),
        base_operator_(base_operator) {}
  ~LiteKernelMod() override = default;

  int Prepare() override;
  int ReSize() override;
  int Run() override;

 public:
  std::string KernelType() const { return base_operator_->name(); }

 private:
  KernelModPtr kernel_mod_;
  CNodePtr cnode_;
  BaseOperatorPtr base_operator_;
};

kernel::KernelExec *FindKernelMod(const CNodePtr &cnode, const BaseOperatorPtr &base_operator,
                                  const std::vector<lite::Tensor *> &in_tensors,
                                  const std::vector<lite::Tensor *> &out_tensors, const lite::InnerContext *ctx);
}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_SRC_LITERT_LITE_KERNEL_MOD_H_
