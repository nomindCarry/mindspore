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

#ifndef MINDSPORE_CORE_OPS_SPARSE_SOFTMAX_H_
#define MINDSPORE_CORE_OPS_SPARSE_SOFTMAX_H_
#include <memory>
#include <vector>
#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameSparseSoftmax = "SparseSoftmax";
/// \brief Similar to softmax but with the catch that the implicitly zero
/// elements do not participate. Refer to Python API @ref
/// mindspore.ops.SparseSoftmax for more details.
class MIND_API SparseSoftmax : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(SparseSoftmax);
  /// \brief Constructor.
  SparseSoftmax() : BaseOperator(kNameSparseSoftmax) { InitIOName({"indices", "values", "shape"}, {"output"}); }
  /// \brief Init.
  void Init() const {}
};

abstract::AbstractBasePtr SparseSoftmaxInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                             const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_SPARSE_SOFTMAX_H_