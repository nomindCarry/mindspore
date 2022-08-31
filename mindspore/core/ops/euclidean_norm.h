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
#ifndef MINDSPORE_CORE_OPS_EUCLIDEANNORM_H_
#define MINDSPORE_CORE_OPS_EUCLIDEANNORM_H_
#include <map>
#include <vector>
#include <string>
#include <memory>

#include "ops/base_operator.h"
#include "mindapi/base/types.h"

namespace mindspore {
namespace ops {
constexpr auto kNameEuclideanNorm = "EuclideanNorm";
class MIND_API EuclideanNorm : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(EuclideanNorm);
  EuclideanNorm() : BaseOperator(kNameEuclideanNorm) { InitIOName({"x", "axes"}, {"y"}); }
  void Init(const bool keep_dims = false);

  void set_keep_dims(const bool keep_dims);

  bool get_keep_dims() const;
  // use to calculate size in kernel
  std::vector<int64_t> get_axes() const;
};

abstract::AbstractBasePtr EuclideanNormInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                             const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_EUCLIDEAN_NORM_H_
