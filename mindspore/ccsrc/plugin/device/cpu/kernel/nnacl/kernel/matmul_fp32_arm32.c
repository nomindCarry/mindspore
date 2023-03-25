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

#ifdef ENABLE_ARM32
#include "nnacl/kernel/matmul_fp32_arm32.h"
#include "nnacl/kernel/matmul_fp32_base.h"

KernelBase *CreateMatmulFp32Arm32() {
  MatmulFp32Struct *matmul = (MatmulFp32Struct *)CreateMatmulFp32();
  return (KernelBase *)matmul;
}
#endif
