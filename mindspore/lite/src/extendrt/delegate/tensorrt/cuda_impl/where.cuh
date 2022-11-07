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

#ifndef MINDSPORE_LITE_SRC_DELEGATE_TENSORRT_CDUA_IMPL_WHERE_H_
#define MINDSPORE_LITE_SRC_DELEGATE_TENSORRT_CDUA_IMPL_WHERE_H_
template <typename T>
void Where(const int *cond, const T *input_x, const T *input_y, T *output, int element_cnt, cudaStream_t stream);
#endif  // MINDSPORE_LITE_SRC_DELEGATE_TENSORRT_CDUA_IMPL_LOGICAL_H_