# Copyright 2020-2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""pooling"""
from __future__ import absolute_import

from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore._checkparam import Rel, Validator as validator
from mindspore.ops.primitive import constexpr
from mindspore.common.tensor import Tensor
import mindspore.context as context
from mindspore.common import dtype as mstype
from mindspore.ops.operations.nn_ops import AdaptiveMaxPool2D
from mindspore.ops.operations.nn_ops import AdaptiveMaxPool3D, AdaptiveAvgPool3D
from mindspore.ops.operations.nn_ops import FractionalMaxPoolWithFixedKsize, FractionalMaxPool3DWithFixedKsize
from mindspore.ops.operations.nn_ops import MaxPool3DWithArgmax
from mindspore.nn.cell import Cell

__all__ = ['AvgPool3d', 'MaxPool3d', 'AvgPool2d', 'MaxPool2d', 'AvgPool1d', 'MaxPool1d', 'FractionalMaxPool2d',
           'FractionalMaxPool3d', 'AdaptiveAvgPool1d', 'AdaptiveMaxPool1d', 'AdaptiveMaxPool2d', 'AdaptiveMaxPool3d',
           'AdaptiveAvgPool2d', 'AdaptiveAvgPool3d']


class _PoolNd(Cell):
    """N-D  AvgPool"""

    def __init__(self, kernel_size, stride, pad_mode, data_format="NCHW"):
        """Initialize _PoolNd."""
        super(_PoolNd, self).__init__()
        validator.check_value_type('pad_mode', pad_mode, [str], self.cls_name)
        self.pad_mode = validator.check_string(pad_mode.upper(), ['VALID', 'SAME'], 'pad_mode', self.cls_name)
        self.format = validator.check_string(data_format, ['NCHW', 'NHWC'], 'format', self.cls_name)
        if context.get_context("device_target") != "GPU" and self.format == "NHWC":
            raise ValueError(f"For '{self.cls_name}, the 'NHWC' format only support in GPU target, but got device "
                             f"target {context.get_context('device_target')}.")

        def _check_int_or_tuple(arg_name, arg_value):
            validator.check_value_type(arg_name, arg_value, [int, tuple], self.cls_name)
            error_msg = f"For '{self.cls_name}', the '{arg_name}' must be an positive int number or " \
                        f"a tuple of two positive int numbers, but got {arg_value}"
            if isinstance(arg_value, int):
                if arg_value <= 0:
                    raise ValueError(error_msg)
            elif len(arg_value) == 2:
                for item in arg_value:
                    if isinstance(item, int) and item > 0:
                        continue
                    raise ValueError(error_msg)
            else:
                raise ValueError(error_msg)
            return arg_value

        self.kernel_size = _check_int_or_tuple('kernel_size', kernel_size)
        self.stride = _check_int_or_tuple('stride', stride)

    def construct(self, *inputs):
        pass

    def extend_repr(self):
        return 'kernel_size={kernel_size}, stride={stride}, pad_mode={pad_mode}'.format(**self.__dict__)


@constexpr
def _shape_check(in_shape, prim_name=None):
    msg_prefix = f"For '{prim_name}', the" if prim_name else "The"
    if len(in_shape) != 3:
        raise ValueError(f"{msg_prefix} input must has 3 dim, but got {len(in_shape)}")


class MaxPool3d(Cell):
    r"""
    3D max pooling operation.

    Applies a 3D max pooling over an input Tensor which can be regarded as a composition of 3D planes.

    Typically the input is of shape :math:`(N_{in}, C_{in}, D_{in}, H_{in}, W_{in})`, MaxPool outputs
    regional maximum in the :math:`(D_{in}, H_{in}, W_{in})`-dimension. Given kernel size is
    :math:`ks = (d_{ker}, h_{ker}, w_{ker})` and stride is :math:`s = (s_0, s_1, s_2)`, the operation is as follows.

    .. math::
        \text{output}(N_i, C_j, d, h, w) =
        \max_{l=0, \ldots, d_{ker}-1} \max_{m=0, \ldots, h_{ker}-1} \max_{n=0, \ldots, w_{ker}-1}
        \text{input}(N_i, C_j, s_0 \times d + l, s_1 \times h + m, s_2 \times w + n)

    Args:
        kernel_size (Union[int, tuple[int]]): The size of kernel used to take the maximum value,
            is an int number that represents depth, height and width of the kernel, or a tuple
            of three int numbers that represent depth, height and width respectively. Default: 1.
        stride (Union[int, tuple[int]]): The moving stride of pooling operation, an int number that represents
            the moving stride of pooling kernel in the directions of depth, height and the width,
            or a tuple of three int numbers that represent depth, height and width of movement respectively. Default: 1.
        padding (Union[int, tuple[int]]): Pooling padding length. An int number that represents the depth,
            height and width of movement are both stride, or a tuple of three int numbers that represent depth,
            height and width of movement respectively.
        dilation (Union[int, tuple[int]]): Control the spacing of elements in the pooling kernel. Default: 1.
        return_indices (bool): If True, output is a Tuple of 2 Tensors, representing the maxpool result and where
            the max values are generated. Otherwise, only the maxpool result is returned.
        ceil_mode (bool): Whether to use ceil or floor to calculate output shape. Default: False.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N_{in}, C_{in}, D_{in}, H_{in}, W_{in})` with data type of int8,
          int16, int32, int64, uint8, uint16, uint32, uint64, float16, float32 or float64.

    Outputs:
        If `return_indices` is False, output is a Tensor, with shape :math:`(N, C, D_{out}, H_{out}, W_{out})`,
        with the same data type as `x`.

        If `return_indices` is True, output is a Tuple of 2 Tensors, representing the maxpool result and where
        the max values are generated.

        - **output** (Tensor) - Maxpooling result, with shape :math:`(N_{out}, C_{out}, D_{out}, H_{out}, W_{out})`.
          It has the same data type as `x`.
        - **argmax** (Tensor) - Index corresponding to the maximum value. Data type is int64.

    Raises:
        TypeError: If `x` is not a Tensor.
        ValueError: If length of shape of `x` is not equal to 5.
        TypeError: If `kernel_size` , `stride` , `padding` or `dilation` is neither an int nor a tuple.
        ValueError: If `kernel_size` or `stride` is less than 1.
        ValueError: If `padding` is less than 0.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> import mindspore as ms
        >>> import mindspore.nn as nn
        >>> import numpy as np
        >>> pool1 = nn.MaxPool3d(kernel_size=3, stride=1, padding=1)
        >>> pool2 = nn.MaxPool3d(kernel_size=3, stride=1, padding=1, return_indices=True)
        >>> x = ms.Tensor(np.random.randint(0, 10, [1, 2, 2, 2, 2]), ms.float32)
        >>> output1 = pool1(x)
        >>> print(output1)
        [[[[[8. 8.]
            [8. 8.]]
           [[8. 8.]
            [8. 8.]]]
          [[[9. 9.]
            [9. 9.]]
           [[9. 9.]
            [9. 9.]]]]]
        >>> output2 = pool2(x)
        >>> print(output2)
        (Tensor(shape=[1, 2, 2, 2, 2], dtype=Float32, value=
        [[[[[8.00000000e+000, 8.00000000e+000],
            [8.00000000e+000, 8.00000000e+000]],
           [[8.00000000e+000, 8.00000000e+000],
            [8.00000000e+000, 8.00000000e+000]]],
          [[[9.00000000e+000, 9.00000000e+000],
            [9.00000000e+000, 9.00000000e+000]],
           [[9.00000000e+000, 9.00000000e+000],
            [9.00000000e+000, 9.00000000e+000]]]]]), Tensor(shape=[1, 2, 2, 2, 2], dtype=Int64, value=
        [[[[[7, 7],
            [7, 7]],
           [[7, 7],
            [7, 7]]],
          [[[2, 2],
            [2, 2]],
           [[2, 2],
            [2, 2]]]]]))
    """

    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        """Initialize MaxPool3d."""
        super(MaxPool3d, self).__init__()
        self.return_indices = return_indices
        self.max_pool = MaxPool3DWithArgmax(kernel_size, stride, padding, dilation, ceil_mode)

    def construct(self, x):
        output_tensor, argmax = self.max_pool(x)
        if self.return_indices:
            return output_tensor, argmax
        return output_tensor


class MaxPool2d(_PoolNd):
    r"""
    2D max pooling operation for temporal data.

    Applies a 2D max pooling over an input Tensor which can be regarded as a composition of 2D planes.

    Typically the input is of shape :math:`(N_{in}, C_{in}, H_{in}, W_{in})`, MaxPool2d outputs
    regional maximum in the :math:`(H_{in}, W_{in})`-dimension. Given kernel size
    :math:`ks = (h_{ker}, w_{ker})` and stride :math:`s = (s_0, s_1)`, the operation is as follows.

    .. math::
        \text{output}(N_i, C_j, h, w) = \max_{m=0, \ldots, h_{ker}-1} \max_{n=0, \ldots, w_{ker}-1}
        \text{input}(N_i, C_j, s_0 \times h + m, s_1 \times w + n)

    Note:
        pad_mode for training only supports "same" and "valid".

    Args:
        kernel_size (Union[int, tuple[int]]): The size of kernel used to take the max value,
            is an int number that represents height and width are both kernel_size,
            or a tuple of two int numbers that represent height and width respectively.
            Default: 1.
        stride (Union[int, tuple[int]]): The distance of kernel moving, an int number that represents
            the height and width of movement are both stride, or a tuple of two int numbers that
            represent height and width of movement respectively. Default: 1.
        pad_mode (str): The optional value for pad mode, is "same" or "valid", not case sensitive.
            Default: "valid".

            - same: Adopts the way of completion. The height and width of the output will be the same as
              the input. The total number of padding will be calculated in horizontal and vertical
              directions and evenly distributed to top and bottom, left and right if possible.
              Otherwise, the last extra padding will be done from the bottom and the right side.

            - valid: Adopts the way of discarding. The possible largest height and width of output
              will be returned without padding. Extra pixels will be discarded.
        data_format (str): The optional value for data format, is 'NHWC' or 'NCHW'.
            Default: 'NCHW'.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, C_{out}, H_{out}, W_{out})`.

    Raises:
        TypeError: If `kernel_size` or `stride` is neither int nor tuple.
        ValueError: If `pad_mode` is neither 'valid' nor 'same' with not case sensitive.
        ValueError: If `data_format` is neither 'NCHW' nor 'NHWC'.
        ValueError: If `kernel_size` or `stride` is less than 1.
        ValueError: If length of shape of `x` is not equal to 4.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> pool = nn.MaxPool2d(kernel_size=3, stride=1)
        >>> x = Tensor(np.random.randint(0, 10, [1, 2, 4, 4]), mindspore.float32)
        >>> output = pool(x)
        >>> print(output.shape)
        (1, 2, 2, 2)
    """

    def __init__(self, kernel_size=1, stride=1, pad_mode="valid", data_format="NCHW"):
        """Initialize MaxPool2d."""
        super(MaxPool2d, self).__init__(kernel_size, stride, pad_mode, data_format)
        self.max_pool = P.MaxPool(kernel_size=self.kernel_size,
                                  strides=self.stride,
                                  pad_mode=self.pad_mode,
                                  data_format=self.format)

    def construct(self, x):
        out = self.max_pool(x)
        return out


class MaxPool1d(_PoolNd):
    r"""
    1D max pooling operation for temporal data.

    Applies a 1D max pooling over an input Tensor which can be regarded as a composition of 1D planes.

    Typically the input is of shape :math:`(N_{in}, C_{in}, L_{in})`, MaxPool1d outputs
    regional maximum in the :math:`(L_{in})`-dimension. Given kernel size
    :math:`ks = (l_{ker})` and stride :math:`s = (s_0)`, the operation is as follows:

    .. math::
        \text{output}(N_i, C_j, l) = \max_{n=0, \ldots, l_{ker}-1}
        \text{input}(N_i, C_j, s_0 \times l + n)

    Note:
        pad_mode for training only supports "same" and "valid".

    Args:
        kernel_size (int): The size of kernel used to take the max value, Default: 1.
        stride (int): The distance of kernel moving, an int number that represents
            the width of movement is stride, Default: 1.
        pad_mode (str): The optional value for pad mode, is "same" or "valid", not case sensitive.
            Default: "valid".

            - same: Adopts the way of completion. The total number of padding will be calculated in horizontal
              and vertical directions and evenly distributed to top and bottom, left and right if possible.
              Otherwise, the last extra padding will be done from the bottom and the right side.

            - valid: Adopts the way of discarding. The possible largest height and width of output
              will be returned without padding. Extra pixels will be discarded.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C, L_{in})`.

    Outputs:
        Tensor of shape :math:`(N, C, L_{out})`.

    Raises:
        TypeError: If `kernel_size` or `strides` is not an int.
        ValueError: If `pad_mode` is neither 'valid' nor 'same' with not case sensitive.
        ValueError: If `data_format` is neither 'NCHW' nor 'NHWC'.
        ValueError: If `kernel_size` or `strides` is less than 1.
        ValueError: If length of shape of `x` is not equal to 3.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> max_pool = nn.MaxPool1d(kernel_size=3, stride=1)
        >>> x = Tensor(np.random.randint(0, 10, [1, 2, 4]), mindspore.float32)
        >>> output = max_pool(x)
        >>> result = output.shape
        >>> print(result)
        (1, 2, 2)
    """

    def __init__(self, kernel_size=1, stride=1, pad_mode="valid"):
        """Initialize MaxPool1d."""
        super(MaxPool1d, self).__init__(kernel_size, stride, pad_mode)
        validator.check_value_type('kernel_size', kernel_size, [int], self.cls_name)
        validator.check_value_type('stride', stride, [int], self.cls_name)
        validator.check_value_type('pad_mode', pad_mode, [str], self.cls_name)
        self.pad_mode = validator.check_string(pad_mode.upper(), ['VALID', 'SAME'], 'pad_mode', self.cls_name)
        validator.check_int(kernel_size, 1, Rel.GE, "kernel_size", self.cls_name)
        validator.check_int(stride, 1, Rel.GE, "stride", self.cls_name)
        self.kernel_size = (1, kernel_size)
        self.stride = (1, stride)
        self.max_pool = P.MaxPool(kernel_size=self.kernel_size,
                                  strides=self.stride,
                                  pad_mode=self.pad_mode)
        self.shape = F.shape
        self.reduce_mean = P.ReduceMean(keep_dims=True)
        self.expand = P.ExpandDims()
        self.squeeze = P.Squeeze(2)

    def construct(self, x):
        _shape_check(self.shape(x), self.cls_name)
        x = self.expand(x, 2)
        output = self.max_pool(x)
        output = self.squeeze(output)
        return output


class AvgPool3d(Cell):
    r"""
    3D average pooling operation.

    Applies a 3D average pooling over an input Tensor which can be regarded as a composition of 3D input planes.
    Typically the input is of shape :math:`(N, C, D_{in}, H_{in}, W_{in})`, and AvgPool3D outputs
    regional average in the :math:`(D_{in}, H_{in}, W_{in})`-dimension. Given kernel size
    is :math:`ks = (d_{ker}, h_{ker}, w_{ker})` and stride :math:`s = (s_0, s_1, s_2)`, the operation is as follows.

    .. warning::
        "kernel_size" is in the range [1, 255]. "stride" is in the range [1, 63].

    .. math::
        \text{output}(N_i, C_j, d, h, w) =
        \frac{1}{d_{ker} * h_{ker} * w_{ker}} \sum_{l=0}^{d_{ker}-1} \sum_{m=0}^{h_{ker}-1} \sum_{n=0}^{w_{ker}-1}
        \text{input}(N_i, C_j, s_0 \times d + l, s_1 \times h + m, s_2 \times w + n)

    Args:
        kernel_size (Union[int, tuple[int]]): The size of kernel used to take the average value,
            is an int number that represents depth, height and width are both kernel_size, or a tuple
            of three int numbers that represent depth, height and width respectively.
        stride (Union[int, tuple[int]]): The distance of kernel moving, an int number that represents
            the depth, height and width of movement are both stride, or a tuple of three int numbers that
            represent depth, height and width of movement respectively. Default: None.
        padding (Union(int, tuple[int])): The padding value to be filled. Default: 0. If `padding` is an integer,
                    the paddings of head, tail, top, bottom, left and right are the same, equal to padding.
                    If `padding` is a tuple of six integers, the padding of head, tail, top, bottom, left and right
                    equal to padding[0], padding[1], padding[2], padding[3], padding[4] and padding[5] correspondingly.
        ceil_mode (bool): If True, use ceil to compute the output shape instead of floor. Default: False.
        count_include_pad (bool): If True, averaging calculation will include the zero-padding. Default: True.
        divisor_override (int): If specified, it will be used as divisor in the averaging calculation,
            otherwise kernel_size will be used. Default: None.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C, D_{in}, H_{in}, W_{in})`.
          Currently support float16 and float32 data type.

    Outputs:
        Tensor, with shape :math:`(N, C, D_{out}, H_{out}, W_{out})`, with the same data type with `x`.

    Raises:
        TypeError: If `kernel_size`, `stride` or `padding` is neither an int nor a tuple.
        TypeError: If `ceil_mode` or `count_include_pad` is not a bool.
        TypeError: If `data_format` is not a string.
        TypeError: If `divisor_override` is not an int.
        ValueError: If numbers in `kernel_size` or `stride` are not positive.
        ValueError: If `kernel_size` or `stride` is a tuple whose length is not equal to 3.
        ValueError: If `padding` is a tuple whose length is not equal to 6.
        ValueError: If element of `padding` is less than 0.
        ValueError: If length of shape of `x` is not equal to 5.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> import mindspore.nn as nn
        >>> import numpy as np
        >>> pool = nn.AvgPool3d(kernel_size=3, stride=1)
        >>> x = ms.Tensor(np.random.randint(0, 10, [1, 2, 4, 4, 5]), ms.float32)
        >>> output = pool(x)
        >>> print(output.shape)
        (1, 2, 2, 2, 3)
    """

    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True,
                 divisor_override=None):
        """Initialize AvgPool3d."""
        super(AvgPool3d, self).__init__()
        stride = stride if (stride is not None) else kernel_size
        if not divisor_override:
            divisor_override = 0
        self.avg_pool = P.AvgPool3D(kernel_size, stride, "pad", padding, ceil_mode, count_include_pad,
                                    divisor_override)

    def construct(self, x):
        out = self.avg_pool(x)
        return out


class AvgPool2d(_PoolNd):
    r"""
    2D average pooling for temporal data.

    Applies a 2D average pooling over an input Tensor which can be regarded as a composition of 2D input planes.

    Typically the input is of shape :math:`(N_{in}, C_{in}, H_{in}, W_{in})`, AvgPool2d outputs
    regional average in the :math:`(H_{in}, W_{in})`-dimension. Given kernel size
    :math:`ks = (h_{ker}, w_{ker})` and stride :math:`s = (s_0, s_1)`, the operation is as follows:

    .. math::
        \text{output}(N_i, C_j, h, w) = \frac{1}{h_{ker} * w_{ker}} \sum_{m=0}^{h_{ker}-1} \sum_{n=0}^{w_{ker}-1}
        \text{input}(N_i, C_j, s_0 \times h + m, s_1 \times w + n)

    Note:
        pad_mode for training only supports "same" and "valid".

    Args:
        kernel_size (Union[int, tuple[int]]): The size of kernel used to take the average value.
            The data type of kernel_size must be int and the value represents the height and width,
            or a tuple of two int numbers that represent height and width respectively.
            Default: 1.
        stride (Union[int, tuple[int]]): The distance of kernel moving, an int number that represents
            the height and width of movement are both strides, or a tuple of two int numbers that
            represent height and width of movement respectively. Default: 1.
        pad_mode (str): The optional value for pad mode, is "same" or "valid", not case sensitive.
            Default: "valid".

            - same: Adopts the way of completion. The height and width of the output will be the same as
              the input. The total number of padding will be calculated in horizontal and vertical
              directions and evenly distributed to top and bottom, left and right if possible.
              Otherwise, the last extra padding will be done from the bottom and the right side.

            - valid: Adopts the way of discarding. The possible largest height and width of output
              will be returned without padding. Extra pixels will be discarded.
        data_format (str): The optional value for data format, is 'NHWC' or 'NCHW'.
            Default: 'NCHW'.


    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, C_{out}, H_{out}, W_{out})`.

    Raises:
        TypeError: If `kernel_size` or `strides` is neither int nor tuple.
        ValueError: If `pad_mode` is neither 'valid' nor 'same' with not case sensitive.
        ValueError: If `data_format` is neither 'NCHW' nor 'NHWC'.
        ValueError: If `kernel_size` or `strides` is less than 1.
        ValueError: If length of shape of `x` is not equal to 4.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> pool = nn.AvgPool2d(kernel_size=3, stride=1)
        >>> x = Tensor(np.random.randint(0, 10, [1, 2, 4, 4]), mindspore.float32)
        >>> output = pool(x)
        >>> print(output.shape)
        (1, 2, 2, 2)
    """

    def __init__(self,
                 kernel_size=1,
                 stride=1,
                 pad_mode="valid",
                 data_format="NCHW"):
        """Initialize AvgPool2d."""
        super(AvgPool2d, self).__init__(kernel_size, stride, pad_mode, data_format)
        self.avg_pool = P.AvgPool(kernel_size=self.kernel_size,
                                  strides=self.stride,
                                  pad_mode=self.pad_mode,
                                  data_format=self.format)

    def construct(self, x):
        return self.avg_pool(x)


class AvgPool1d(_PoolNd):
    r"""
    1D average pooling for temporal data.

    Applies a 1D average pooling over an input Tensor which can be regarded as a composition of 1D input planes.

    Typically the input is of shape :math:`(N_{in}, C_{in}, L_{in})`, AvgPool1d outputs
    regional average in the :math:`(L_{in})`-dimension. Given kernel size
    :math:`ks = l_{ker}` and stride :math:`s = s_0`, the operation is as follows:

    .. math::
        \text{output}(N_i, C_j, l) = \frac{1}{l_{ker}} \sum_{n=0}^{l_{ker}-1}
        \text{input}(N_i, C_j, s_0 \times l + n)

    Note:
        pad_mode for training only supports "same" and "valid".

    Args:
        kernel_size (int): The size of kernel window used to take the average value, Default: 1.
        stride (int): The distance of kernel moving, an int number that represents
            the width of movement is strides, Default: 1.
        pad_mode (str): The optional value for pad mode, is "same" or "valid", not case sensitive.
            Default: "valid".

            - same: Adopts the way of completion. The height and width of the output will be the same as
              the input. The total number of padding will be calculated in horizontal and vertical
              directions and evenly distributed to top and bottom, left and right if possible.
              Otherwise, the last extra padding will be done from the bottom and the right side.

            - valid: Adopts the way of discarding. The possible largest height and width of output
              will be returned without padding. Extra pixels will be discarded.


    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, L_{in})`.

    Outputs:
        Tensor of shape :math:`(N, C_{out}, L_{out})`.

    Raises:
        TypeError: If `kernel_size` or `stride` is not an int.
        ValueError: If `pad_mode` is neither 'same' nor 'valid' with not case sensitive.
        ValueError: If `kernel_size` or `strides` is less than 1.
        ValueError: If length of shape of `x` is not equal to 3.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> pool = nn.AvgPool1d(kernel_size=6, stride=1)
        >>> x = Tensor(np.random.randint(0, 10, [1, 3, 6]), mindspore.float32)
        >>> output = pool(x)
        >>> result = output.shape
        >>> print(result)
        (1, 3, 1)
    """

    def __init__(self,
                 kernel_size=1,
                 stride=1,
                 pad_mode="valid"):
        """Initialize AvgPool1d."""
        validator.check_value_type('kernel_size', kernel_size, [int], self.cls_name)
        validator.check_value_type('stride', stride, [int], self.cls_name)
        validator.check_value_type('pad_mode', pad_mode, [str], self.cls_name)
        self.pad_mode = validator.check_string(pad_mode.upper(), ['VALID', 'SAME'], 'pad_mode', self.cls_name)
        validator.check_int(kernel_size, 1, Rel.GE, "kernel_size", self.cls_name)
        validator.check_int(stride, 1, Rel.GE, "stride", self.cls_name)
        super(AvgPool1d, self).__init__(kernel_size, stride, pad_mode)
        self.kernel_size = (1, kernel_size)
        self.stride = (1, stride)
        self.avg_pool = P.AvgPool(kernel_size=self.kernel_size,
                                  strides=self.stride,
                                  pad_mode=self.pad_mode)
        self.shape = F.shape
        self.reduce_mean = P.ReduceMean(keep_dims=True)
        self.slice = P.Slice()
        self.expand = P.ExpandDims()
        self.squeeze = P.Squeeze(2)

    def construct(self, x):
        x = F.depend(x, _shape_check(self.shape(x), self.cls_name))
        batch, channel, width = self.shape(x)
        if width == self.kernel_size[1]:
            x = self.reduce_mean(x, 2)
        elif width - self.kernel_size[1] < self.stride[1]:
            x = self.slice(x, (0, 0, 0), (batch, channel, self.kernel_size[1]))
            x = self.reduce_mean(x, 2)
        else:
            x = self.expand(x, 2)
            x = self.avg_pool(x)
            x = self.squeeze(x)
        return x


@constexpr
def _adaptive_shape_check(in_shape, output_size, prim_name):
    """Check shape."""
    msg_prefix = "For {}, the".format(prim_name)
    if len(in_shape) != 3:
        raise ValueError("{} input must has 3 dim, but got {}.".format(msg_prefix, len(in_shape)))
    if in_shape[2] < output_size:
        raise ValueError("{} input's last dimension must be greater or equal to "
                         "output size {}, but got {}.".format(msg_prefix, output_size, in_shape[2]))
    if in_shape[2] % output_size != 0:
        raise ValueError("{} input's last dimension must be divisible by "
                         "output size {}, but got {}.".format(msg_prefix, output_size, in_shape[2]))


@constexpr
def _adaptive_dtype_check(x_dtype, prim_name):
    """Check dtype."""
    if x_dtype not in [mstype.float16, mstype.float32]:
        raise TypeError("For {}, the x_dtype must be float16 or float32, "
                        "but got {}.".format(prim_name, x_dtype))


class AdaptiveAvgPool1d(Cell):
    r"""
    1D adaptive average pooling for temporal data.

    Applies a 1D adaptive average pooling over an input Tensor which can be regarded as
    a composition of 1D input planes.

    Typically, the input is of shape :math:`(N_{in}, C_{in}, L_{in})`,
    AdaptiveAvgPool1d outputs regional average in the :math:`L_{in}`-dimension.
    The output is of shape :math:`(N_{in}, C_{in}, L_{out})`,
    where :math:`L_{out}` is defined by `output_size`.

    Note:
        :math:`L_{in}` must be divisible by `output_size`.

    Args:
        output_size (int): the target output size :math:`L_{out}`.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, L_{in})`, with float16 or float32 data type.

    Outputs:
        Tensor of shape :math:`(N, C_{in}, L_{out})`, has the same type as `x`.

    Raises:
        TypeError: If `output_size` is not an int.
        TypeError: If `x` is neither float16 nor float32.
        ValueError: If `output_size` is less than 1.
        ValueError: If length of shape of `x` is not equal to 3.
        ValueError: If the last dimension of `x` is smaller than `output_size`.
        ValueError: If the last dimension of `x` is not divisible by `output_size`.


    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, nn
        >>> import numpy as np
        >>> pool = nn.AdaptiveAvgPool1d(output_size=2)
        >>> x = Tensor(np.random.randint(0, 10, [1, 3, 6]), mindspore.float32)
        >>> output = pool(x)
        >>> result = output.shape
        >>> print(result)
        (1, 3, 2)
    """

    def __init__(self, output_size):
        """Initialize AdaptiveAvgPool1d."""
        super(AdaptiveAvgPool1d, self).__init__()
        validator.check_value_type('output_size', output_size, [int], self.cls_name)
        validator.check_int(output_size, 1, Rel.GE, "output_size", self.cls_name)
        self.shape = F.shape
        self.expand = P.ExpandDims()
        self.squeeze = P.Squeeze(2)
        self.output_size = output_size
        self.dtype = P.DType()

    def construct(self, x):
        _adaptive_shape_check(self.shape(x), self.output_size, self.cls_name)
        _adaptive_dtype_check(self.dtype(x), self.cls_name)

        _, _, width = self.shape(x)
        stride = width // self.output_size
        kernel_size = width - (self.output_size - 1) * stride

        stride = (1, width // self.output_size)
        kernel_size = (1, kernel_size)

        x = self.expand(x, 2)
        avg_pool = P.AvgPool(kernel_size=kernel_size, strides=stride)
        x = avg_pool(x)
        x = self.squeeze(x)

        return x


class AdaptiveAvgPool2d(Cell):
    r"""
    2D adaptive average pooling for temporal data.

    This operator applies a 2D adaptive average pooling to an input signal composed of multiple input planes.
    That is, for any input size, the size of the specified output is H x W.
    The number of output features is equal to the number of input features.

    The input and output data format can be "NCHW" and "CHW". N is the batch size, C is the number of channels,
    H is the feature height, and W is the feature width.

    .. math::
        \begin{align}
        h_{start} &= floor(i * H_{in} / H_{out})\\
        h_{end} &= ceil((i + 1) * H_{in} / H_{out})\\
        w_{start} &= floor(j * W_{in} / W_{out})\\
        w_{end} &= ceil((j + 1) * W_{in} / W_{out})\\
        Output(i,j) &= \frac{\sum Input[h_{start}:h_{end}, w_{start}:w_{end}]}{(h_{end}- h_{start})
        * (w_{end}- w_{start})}
        \end{align}

    Args:
        output_size (Union[int, tuple]): The target output size is H x W.
            `ouput_size` can be a tuple consisted of int type H and W, or a single H for H x H, or None.
            If it is None, it means the output size is the same as the input size.

    Inputs:
        - **x** (Tensor) - The input of AdaptiveAvgPool2d, which is a 3D or 4D tensor,
          with float16, float32 or float64 data type.

    Outputs:
        Tensor of shape :math:`(N, C_{out}, H_{out}, W_{out})`.

    Raises:
        ValueError: If `output_size` is a tuple and the length of `output_size` is not 2.
        TypeError: If `x` is not a Tensor.
        TypeError: If dtype of `x` is not float16, float32 or float64.
        ValueError: If the dimension of `x` is less than or equal to the dimension of `output_size`.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> pool = nn.AdaptiveAvgPool2d(2)
        >>> input_x = Tensor(np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        ...                            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        ...                            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]), mindspore.float32)
        >>> output = pool(input_x)
        >>> result = output.shape
        >>> print(result)
        (3, 2, 2)
    """

    def __init__(self, output_size):
        """Initialize AdaptiveAvgPool2d."""
        super(AdaptiveAvgPool2d, self).__init__()
        self.adaptive_avgpool2d = P.AdaptiveAvgPool2D(output_size)

    def construct(self, x):
        return self.adaptive_avgpool2d(x)


class AdaptiveAvgPool3d(Cell):
    r"""
    3D adaptive average pooling for temporal data.

    This operator applies a 3D adaptive average pooling to an input signal composed of multiple input planes.
    That is, for any input size, the size of the specified output is :math:`(D, H, W)`.
    The number of output features is equal to the number of input planes.

    Suppose the last 3 dimension size of x is :math:`(inD, inH, inW)`, the last 3 dimension size of output is
    :math:`(outD, outH, outW)`.

    .. math::
        \begin{array}{ll} \\
            \forall \quad od \in [0,outD-1], oh \in [0,outH-1], ow \in [0,outW-1]\\
            output[od,oh,ow] = \\
            \qquad mean(x[istartD:iendD+1,istartH:iendH+1,istartW:iendW+1])\\
            where,\\
            \qquad istartD= \left\lceil \frac{od * inD}{outD} \right\rceil \\
            \qquad iendD=\left\lfloor \frac{(od+1)* inD}{outD} \right\rfloor \\
            \qquad istartH=\left\lceil \frac{oh * inH}{outH} \right\rceil \\
            \qquad iendH=\left\lfloor \frac{(oh+1) * inH}{outH} \right\rfloor \\
            \qquad istartW=\left\lceil \frac{ow * inW}{outW} \right\rceil \\
            \qquad iendW=\left\lfloor \frac{(ow+1) * inW}{outW} \right\rfloor
        \end{array}

    Args:
        output_size (Union[int, tuple]): he target output size. `ouput_size` can be a tuple :math:`(D, H, W)`,
            or an int D for :math:`(D, D, D)`. :math:`(D)`, :math:`(H)` and :math:`(W)` can be int or None
            which means the output size is the same as that of the input.

    Inputs:
        - **x** (Tensor) - The input of AdaptiveAvgPool3d, which is a 5D or 4D Tensor.

    Outputs:
        Tensor, with the same type as the `x`.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If dtype of `x` is not float16, float32 or float64.
        ValueError: If the dimension of `x` is not 4D or 5D.
        ValueError: If `output_size` value is not positive.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> # case 1: output_size=(3, 3, 4)
        >>> output_size=(3, 3, 4)
        >>> input_x_val = np.random.randn(4, 3, 5, 6, 7)
        >>> input_x = Tensor(input_x_val, mindspore.float32)
        >>> net = nn.AdaptiveAvgPool3d(output_size)
        >>> output = net(input_x)
        >>> print(output.shape)
        (4, 3, 3, 3, 4)
        >>> # case 2: output_size=4
        >>> output_size=5
        >>> input_x_val = np.random.randn(2, 3, 8, 6, 12)
        >>> input_x = Tensor(input_x_val, mindspore.float32)
        >>> net = nn.AdaptiveAvgPool3d(output_size)
        >>> output = net(input_x)
        >>> print(output.shape)
        (2, 3, 5, 5, 5)
        >>> # case 3: output_size=(None, 4, 5)
        >>> output_size=(None, 4, 5)
        >>> input_x_val = np.random.randn(4, 1, 9, 10, 8)
        >>> input_x = Tensor(input_x_val, mindspore.float32)
        >>> net = nn.AdaptiveAvgPool3d(output_size)
        >>> output = net(input_x)
        >>> print(output.shape)
        (4, 1, 9, 4, 5)
    """

    def __init__(self, output_size):
        """Initialize AdaptiveAvgPool3d."""
        super(AdaptiveAvgPool3d, self).__init__()
        self.adaptive_avg_pool3d = AdaptiveAvgPool3D(output_size)

    def construct(self, x):
        return self.adaptive_avg_pool3d(x)


class AdaptiveMaxPool1d(Cell):
    r"""
    1D adaptive maximum pooling for temporal data.

    Applies a 1D adaptive maximum pooling over an input Tensor which can be regarded as
    a composition of 1D input planes.

    Typically, the input is of shape :math:`(N_{in}, C_{in}, L_{in})`,
    AdaptiveMaxPool1d outputs regional maximum in the :math:`L_{in}`-dimension. The output is of
    shape :math:`(N_{in}, C_{in}, L_{out})`, where :math:`L_{out}` is defined by `output_size`.

    Note:
        :math:`L_{in}` must be divisible by `output_size`.

    Args:
        output_size (int): the target output size :math:`L_{out}`.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, L_{in})`, with float16 or float32 data type.

    Outputs:
        Tensor of shape :math:`(N, C_{in}, L_{out})`, has the same type as `x`.

    Raises:
        TypeError: If `x` is neither float16 nor float32.
        TypeError: If `output_size` is not an int.
        ValueError: If `output_size` is less than 1.
        ValueError: If the last dimension of `x` is smaller than `output_size`.
        ValueError: If the last dimension of `x` is not divisible by `output_size`.
        ValueError: If length of shape of `x` is not equal to 3.


    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, nn
        >>> import numpy as np
        >>> pool = nn.AdaptiveMaxPool1d(output_size=3)
        >>> x = Tensor(np.random.randint(0, 10, [1, 3, 6]), mindspore.float32)
        >>> output = pool(x)
        >>> result = output.shape
        >>> print(result)
        (1, 3, 3)
    """

    def __init__(self, output_size):
        """Initialize AdaptiveMaxPool1d."""
        super(AdaptiveMaxPool1d, self).__init__()
        validator.check_int(output_size, 1, Rel.GE, "output_size", self.cls_name)
        validator.check_value_type('output_size', output_size, [int], self.cls_name)
        self.expand = P.ExpandDims()
        self.squeeze = P.Squeeze(2)
        self.output_size = output_size
        self.shape = F.shape
        self.dtype = P.DType()

    def construct(self, x):
        _adaptive_shape_check(self.shape(x), self.output_size, self.cls_name)
        _adaptive_dtype_check(self.dtype(x), self.cls_name)

        _, _, width = self.shape(x)
        stride = width // self.output_size
        kernel_size = width - (self.output_size - 1) * stride

        stride = (1, width // self.output_size)
        kernel_size = (1, kernel_size)

        max_pool = P.MaxPool(kernel_size=kernel_size, strides=stride)
        x = self.expand(x, 2)
        x = max_pool(x)
        x = self.squeeze(x)

        return x


class AdaptiveMaxPool2d(Cell):
    r"""
    AdaptiveMaxPool2d operation.

    This operator applies a 2D adaptive max pooling to an input signal composed of multiple input planes.
    That is, for any input size, the size of the specified output is H x W.
    The number of output features is equal to the number of input planes.

    The input and output data format can be "NCHW" and "CHW". N is the batch size, C is the number of channels,
    H is the feature height, and W is the feature width.

    For max adaptive pool2d:

    .. math::

        \begin{align}
        h_{start} &= floor(i * H_{in} / H_{out})\\
        h_{end} &= ceil((i + 1) * H_{in} / H_{out})\\
        w_{start} &= floor(j * W_{in} / W_{out})\\
        w_{end} &= ceil((j + 1) * W_{in} / W_{out})\\
        Output(i,j) &= {\max Input[h_{start}:h_{end}, w_{start}:w_{end}]}
        \end{align}

    Note:
        Ascend platform only supports float16 type for input_x.

    Args:
        output_size (Union[int, tuple]): The target output size is H x W.
            ouput_size can be a tuple, or a single H for H x H, and H and W can be int or None
            which means the output size is the same as the input.

        return_indices (bool): If `return_indices` is True, the indices of max value would be output.
            Default: False.

    Inputs:
        - **input_x** (Tensor) - The input of AdaptiveMaxPool2d, which is a 3D or 4D tensor,
          with float16, float32 or float64 data type.

    Outputs:
        Tensor, with the same type as the `input_x`.

        Shape of the output is `input_x_shape[:len(input_x_shape) - len(out_shape)] + out_shape`.

    Raises:
        TypeError: If `output_size` is not int or tuple.
        TypeError: If `input_x` is not a tensor.
        TypeError: If `return_indices` is not a bool.
        TypeError: If dtype of `input_x` is not float16, float32 or float64.
        ValueError: If `output_size` is a tuple and the length of `output_size` is not 2.
        ValueError: If the dimension of `input_x` is not NCHW or CHW.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> # case 1: output_size=(None, 2)
        >>> input_x = Tensor(np.array([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        ...                             [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        ...                             [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]]), mindspore.float32)
        >>> adaptive_max_pool_2d = nn.AdaptiveMaxPool2d((None, 2))
        >>> output = adaptive_max_pool_2d(input_x)
        >>> print(output)
        [[[[2. 3.]
           [5. 6.]
           [8. 9.]]
          [[2. 3.]
           [5. 6.]
           [8. 9.]]
          [[2. 3.]
           [5. 6.]
           [8. 9.]]]]
        >>> # case 2: output_size=2
        >>> adaptive_max_pool_2d = nn.AdaptiveMaxPool2d(2)
        >>> output = adaptive_max_pool_2d(input_x)
        >>> print(output)
        [[[[5. 6.]
           [8. 9.]]
          [[5. 6.]
           [8. 9.]]
          [[5. 6.]
           [8. 9.]]]]
        >>> # case 3: output_size=(1, 2)
        >>> adaptive_max_pool_2d = nn.AdaptiveMaxPool2d((1, 2))
        >>> output = adaptive_max_pool_2d(input_x)
        >>> print(output)
        [[[[8. 9.]]
          [[8. 9.]]
          [[8. 9.]]]]
    """

    def __init__(self, output_size, return_indices=False):
        """Initialize AdaptiveMaxPool2d."""
        super(AdaptiveMaxPool2d, self).__init__()
        self.adaptive_max_pool2d = AdaptiveMaxPool2D(output_size, return_indices)

    def construct(self, input_x):
        return self.adaptive_max_pool2d(input_x)


class AdaptiveMaxPool3d(Cell):
    r"""
    Applies a 3D adaptive max pooling over an input signal composed of several input planes.

    The output is of size :math:`(D, H, W)`, for any input size.
    The number of output features is equal to the number of input planes.

    Args:
        output_size (Union[int, tuple]): The target output size is :math:`(D, H, W)`.
            `ouput_size` can be a tuple with 3 elements, or a single D for :math:`(D, D, D)`. :math:`D`,
            :math:`H` and :math:`W` can be int or None which means the output size is the same as that of
            the input.
        return_indices (bool): If `return_indices` is True, the indices of max value would be output.
            Default: False.

    Inputs:
        - **x** (Tensor) - It is a 4D or 5D Tensor with int8, int16, int32, int64, uint8, uint16, uint32,
          uint64, float16, float32 or float64 data type.

    Outputs:
        - **y** (Tensor) - A Tensor, with the same number of dims and data type as the `x`.
        - **argmax** (Tensor) - The indices along with the outputs, which is a Tensor, with the same shape as the
          `y` and int32 data type. It will output only when `return_indices` is True.

    Raises:
        TypeError: If `x` is not a Tensor.
        ValueError: If the dimensions number of `x` is not 4 or 5.
        TypeError: If dtype of `x` is not int8, int16, int32, int64, uint8, uint16, uint32, uint64,
                   float16, float32 or float64.
        ValueError: If `output_size` is neither an int nor a tuple with shape (3,).

    Supported Platforms:
        ``GPU``

    Examples:
        >>> x = Tensor(np.arange(0,36).reshape((1, 3, 3, 4)).astype(np.float32))
        >>> output_size = (1, 1, 2)
        >>> net = nn.AdaptiveMaxPool3d(output_size, True)
        >>> output = net(x)
        >>> print(output[0].asnumpy())
        [[[[33. 35.]]]]
        >>> print(output[1].asnumpy())
        [[[[33 35]]]]
    """

    def __init__(self, output_size, return_indices=False):
        """Initialize AdaptiveMaxPool3d."""
        super(AdaptiveMaxPool3d, self).__init__()
        self.output_size = Tensor(output_size, dtype=mstype.int32)
        self.return_indices = return_indices
        self.adaptive_max_pool3d = AdaptiveMaxPool3D()

    def construct(self, x):
        output = self.adaptive_max_pool3d(x, self.output_size)
        if self.return_indices:
            return output
        return output[0]


class FractionalMaxPool2d(Cell):
    r"""
    2D fractional max pooling operation for temporal data.

    Applies a 2D fractional max pooling to an input signal composed of multiple input planes.
    The max-pooling operation is applied in kH × kW regions by a stochastic step size determined by
    the target output size. For any input size, the size of the specified output is H x W. The number
    of output features is equal to the number of input planes.

    Fractional MaxPooling is described in the paper `Fractional Max-Pooling <https://arxiv.org/pdf/1412.6071>`_.

    Args:
        kernel_size (Union[int, tuple[int]]): The size of kernel window used to take the maximum value.
            The target `kernel_size` is H x W. `kernel_size` can be a tuple, or a single K for K x K.
            specifying the window size (H, W) of the input tensor.
        output_size (Union[int, tuple[int]]): The target output size is H x W.
            `output_size` can be a tuple, or a single H for H x H.
            specifying the size (H, W) of the output tensor.
            Default: None.
        output_ratio (Union[float, tuple]): The target `output_ratio` is H x W.
            `output_ratio` can be a tuple, or a single H for H x H.
            Specifying the size of the output tensor by using a ratio of the input size.
            Data type : float16, float32, double, and value is between (0, 1).
            Default: None.
        return_indices (bool): If `return_indices` is True, the indices of max value would be output.
            Default: False.
        _random_samples (Tensor): The random step of FractionalMaxPool2d, which is a 3D tensor.
            Tensor of data type : float16, float32, double, and value is between (0, 1).
            Supported shape :math:`(N, C, 2)`.
            Default: None.

    Inputs:
        - **input_x** (Tensor) - Tensor of shape :math:`(N, C, H_{in}, W_{in})`,
          with float16, float32, float64, int32, int64 data type.

    Outputs:
        - **y** (Tensor) - Has the same type as the `input_x`.
          Has the shape :math:`(N, C, output\underline{~}shape{H}, output\underline{~}shape{W})`.

        - **argmax** (Tensor) - The indices along with the outputs, which is a Tensor, with the same shape as the
          `y` and int64 data type. It will output only when `return_indices` is True.

    Raises:
        TypeError: If data type of `input_x` is not one of the following: float16, float32, float64, int32, int64.
        TypeError: If data type of `_random_samples` is not one of the following: float16, float32, float64.
        ValueError: If `kernel_size` is not a number and `kernel_size` is not a tuple of length 2.
        ValueError: If `output_size` is not a number and `output_size` is not a tuple of length 2.
        ValueError: If the sum of `kernel_size` , `output_size` and -1 is larger than the corresponding
                    dimension of `input_x`.
        ValueError: If the dimension of `_random_samples` is not 3.
        ValueError: if `output_size` and `output_ratio` are None at the same time.
        ValueError: If the first dimension size of `input_x` and `_random_samples` is not equal.
        ValueError: If the second dimension size of `input_x` and `_random_samples` is not equal.
        ValueError: If the third dimension size of `_random_samples` is not 2.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> # the kernel_size is an int number and the output_size is a tuple.
        >>> import numpy as np
        >>> from mindspore import nn
        >>> from mindspore import Tensor
        >>> import mindspore.common.dtype as mstype
        >>> input_x = Tensor(np.array([0.3220, 0.9545, 0.7879, 0.0975, 0.3698,
        ...                            0.5135, 0.5740, 0.3435, 0.1895, 0.8764,
        ...                            0.9581, 0.4760, 0.9014, 0.8522, 0.3664,
        ...                            0.4980, 0.9673, 0.9879, 0.6988, 0.9022,
        ...                            0.9304, 0.1558, 0.0153, 0.1559, 0.9852]).reshape([1, 1, 5, 5]), mstype.float32)
        >>> _random_samples = Tensor(np.array([[[0.8, 0.8]]]), mstype.float32)
        >>> net = nn.FractionalMaxPool2d(kernel_size=2, output_size=(2, 2), _random_samples=_random_samples,
        ...                              return_indices=True)
        >>> y, argmax = net(input_x)
        >>> print(y)
        Tensor(shape=[1, 1, 2, 2], dtype=Float32, value=
        [[[[9.54500020e-001, 8.76399994e-001],
           [9.67299998e-001, 9.85199988e-001]]]])
        >>> print(argmax)
        Tensor(shape=[1, 1, 2, 2], dtype=Int64, value=
        [[[[ 1,  9],
           [16, 24]]]])
        >>> net = nn.FractionalMaxPool2d(kernel_size=2, output_ratio=(0.5, 0.5), _random_samples=_random_samples,
        ...                              return_indices=True)
        >>> y, argmax = net(input_x)
        >>> print(y)
        Tensor(shape=[1, 1, 2, 2], dtype=Float32, value=
        [[[[9.54500020e-001, 8.76399994e-001],
           [9.67299998e-001, 9.85199988e-001]]]])
        >>> print(argmax)
        Tensor(shape=[1, 1, 2, 2], dtype=Int64, value=
        [[[[ 1,  9],
           [16, 24]]]])
    """

    def __init__(self, kernel_size, output_size=None, output_ratio=None, return_indices=False, _random_samples=None):
        """Initialize FractionalMaxPool2d."""
        super(FractionalMaxPool2d, self).__init__()
        self.return_indices = return_indices
        self.output_ratio = None
        if _random_samples is None:
            _random_samples = Tensor(np.array([[[0, 0]]]), mstype.float32)
        self.random_samples = _random_samples
        if output_ratio is not None:
            if isinstance(output_ratio, float):
                output_ratio = (output_ratio, output_ratio)
            validator.check_float_range(output_ratio[0], 0.0, 1.0, Rel.INC_RIGHT)
            validator.check_float_range(output_ratio[1], 0.0, 1.0, Rel.INC_RIGHT)
            self.kernel_size = kernel_size
            self.output_ratio = output_ratio
        elif output_size is not None:
            self.fractional_max_pool2d = FractionalMaxPoolWithFixedKsize(kernel_size, output_size)
        else:
            raise ValueError("'output_size' and 'output_ratio' can not be None at the same time.")

    def construct(self, x):
        if self.output_ratio is not None:
            output_size = (int(x.shape[-2] * self.output_ratio[0]), int(x.shape[-1] * self.output_ratio[1]))
            fractional_max_pool2d = FractionalMaxPoolWithFixedKsize(self.kernel_size, output_size)
            output = fractional_max_pool2d(x, self.random_samples)
            if self.return_indices:
                return output
            return output[0]
        output = self.fractional_max_pool2d(x, self.random_samples)
        if self.return_indices:
            return output
        return output[0]


class FractionalMaxPool3d(Cell):
    r"""
    3D fractional max pooling operation for temporal data.

    This operator applies a 3D fractional max pooling over an input signal composed of several input planes.
    The max-pooling operation is applied in kD x kH x kW regions by a stochastic step size determined
    by the target output size.The number of output features is equal to the number of input planes.

    Refer to the paper `Fractional MaxPooling by Ben Graham <https://arxiv.org/abs/1412.6071>`_  for more details.

    The input and output data format can be "NCDHW". N is the batch size, C is the number of channels,
    D the feature depth, H is the feature height, and W is the feature width.

    Args:
        kernel_size (Union[float, tuple]): The target `kernel_size` is D x H x W.
            `kernel_size` can be a tuple, or a single K for K x K x K.
            specifying the window size (D, H, W) of the input tensor.
        output_size (Union[int, tuple]): The target `output_size` is D x H x W.
            `output_size` can be a tuple, or a single H for H x H x H.
            Specifying the size (D, H, W) of the output tensor.
            Default: None.
        output_ratio (Union[float, tuple]): The target `output_ratio` is D x H x W.
            `output_ratio` can be a tuple, or a single H for H x H x H.
            Specifying the size of the output tensor by using a ratio of the input size.
            Data type : float16, float32, double, and value is between (0, 1).
            Default: None.
        return_indices (bool): If `return_indices` is True, the indices of max value would be output.
            Default: False.
        _random_samples (Tensor): The random step of FractionalMaxPool3d, which is a 3D tensor.
            Tensor of data type : float16, float32, double, and value is between (0, 1).
            Supported shape :math:`(N, C, 3)`

    Inputs:
        - **imput_x** (Tensor) - The input of FractionalMaxPool3d, which is a 4D or 5D tensor.
          Tensor of data type : float16, float32, double, int32, int64.
          Supported shape :math:`(N, C, D_{in}, H_{in}, W_{in})` .

    Outputs:
        - **y** (Tensor) - A tensor, the output of FractionalMaxPool3d.
          Has the same data type with `imput_x`.
          Tensor of shape :math:`(N, C, D_{out}, H_{out}, W_{out})` .

        - **argmax** (Tensor) - The indices along with the outputs, which is a Tensor, with the same shape as the
          `y` and int32 data type. It will output only when `return_indices` is True.

    Raises:
        TypeError: If `input_x` is not a 4D or 5D tensor.
        TypeError: If `_random_samples` is not a 3D tensor.
        TypeError: If data type of `imput_x` is not float16, float32, double, int32, int64.
        TypeError: If dtype of `_random_samples` is not float16, float32, double.
        TypeError: If dtype of `argmax` is not int32, int64.
        ValueError: If `output_size` is a tuple and if `output_size` length is not 3.
        ValueError: If `kernel_size` is a tuple and if `kernel_size` length is not 3.
        ValueError: If numbers in `output_size` or `kernel_size` is not positive.
        ValueError: if `output_size` and `output_ratio` are None at the same time.
        ValueError: If the first dimension size of `input_x` and `_random_samples` is not equal.
        ValueError: If the second dimension size of `input_x` and `_random_samples` is not equal.
        ValueError: If the third dimension size of `_random_samples` is not 3.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import nn
        >>> from mindspore import Tensor
        >>> import mindspore.common.dtype as mstype
        >>> x = Tensor(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
        ...            .reshape([1, 1, 2, 2, 4]), mstype.float32)
        >>> _random_samples = Tensor(np.array([0.7, 0.7, 0.7]).reshape([1, 1, 3]), mstype.float32)
        >>> net = nn.FractionalMaxPool3d(kernel_size=(1.0, 1.0, 1.0), output_size=(1, 1, 3),
        ...                              _random_samples=_random_samples, return_indices=True)
        >>> output, argmax = net(x)
        >>> print(output)
        Tensor(shape=[1, 1, 1, 1, 3], dtype=Float32, value=
        [[[[[1.30000000e+001, 1.40000000e+001, 1.60000000e+001]]]]])
        >>> print(argmax)
        Tensor(shape=[1, 1, 1, 1, 3], dtype=Int64, value=
        [[[[[12, 13, 15]]]]])
        >>> net = nn.FractionalMaxPool3d(kernel_size=(1.0, 1.0, 1.0), output_ratio=(0.5, 0.5, 0.5),
        ...                              _random_samples=_random_samples, return_indices=True)
        >>> output, argmax = net(x)
        >>> print(output)
        Tensor(shape=[1, 1, 1, 1, 2], dtype=Float32, value=
        [[[[[1.30000000e+001, 1.60000000e+001]]]]])
        >>> print(argmax)
        Tensor(shape=[1, 1, 1, 1, 2], dtype=Int64, value=
        [[[[[12, 15]]]]])
    """

    def __init__(self, kernel_size, output_size=None, output_ratio=None, return_indices=False, _random_samples=None):
        """Initialize FractionalMaxPool3d."""
        super(FractionalMaxPool3d, self).__init__()
        self.return_indices = return_indices
        self.output_ratio = None
        if _random_samples is None:
            _random_samples = Tensor(np.array([0, 0, 0]).reshape([1, 1, 3]), mstype.float32)
        self.random_samples = _random_samples
        if output_ratio is not None:
            if isinstance(output_ratio, float):
                output_ratio = (output_ratio, output_ratio, output_ratio)
            validator.check_float_range(output_ratio[0], 0.0, 1.0, Rel.INC_RIGHT)
            validator.check_float_range(output_ratio[1], 0.0, 1.0, Rel.INC_RIGHT)
            validator.check_float_range(output_ratio[2], 0.0, 1.0, Rel.INC_RIGHT)
            self.kernel_size = kernel_size
            self.output_ratio = output_ratio
        elif output_size is not None:
            self.fractional_max_pool3d = FractionalMaxPool3DWithFixedKsize(kernel_size, output_size)
        else:
            raise ValueError("'output_size' and 'output_ratio' can not be None at the same time.")

    def construct(self, x):
        if self.output_ratio:
            output_size = (int(x.shape[-3] * self.output_ratio[0]), int(x.shape[-2] * self.output_ratio[1]),
                           int(x.shape[-1] * self.output_ratio[2]))
            fractional_max_pool3d = FractionalMaxPool3DWithFixedKsize(self.kernel_size, output_size)
            output = fractional_max_pool3d(x, self.random_samples)
            if self.return_indices:
                return output
            return output[0]
        output = self.fractional_max_pool3d(x, self.random_samples)
        if self.return_indices:
            return output
        return output[0]
