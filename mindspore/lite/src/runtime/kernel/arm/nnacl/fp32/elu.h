/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_NNACL_FP32_ELU_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_NNACL_FP32_ELU_H_

#include "src/runtime/kernel/arm/nnacl/op_base.h"

struct EluParameter {
  OpParameter op_parameter_;
  float alpha_;
  int thread_num_;
  int in_size_;
};

int Elu(float *input_data, float *output_data, EluParameter *parameter, int task_id);

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_NNACL_FP32_ELU_H_
