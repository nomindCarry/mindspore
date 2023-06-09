mindspore.ops.threshold
=======================

.. py:function:: mindspore.ops.threshold(input_x, thr, value)

    threshold激活函数，按元素计算输出。

    threshold定义为：

    .. math::
        y =
        \begin{cases}
        x, &\text{ if } x > \text{thr} \\
        \text{value}, &\text{ otherwise }
        \end{cases}

    参数：
        - **input_x** (Tensor) - 输入Tensor，数据类型为float16或float32。
        - **thr** (Union[int, float]) - 阈值。
        - **value** (Union[int, float]) - 输入Tensor中element小于阈值时的填充值。

    返回：
        Tensor，数据类型和shape与 `input_x` 的相同。

    异常：
        - **TypeError** - `input_x` 不是Tensor。
        - **TypeError** - `thr` 不是浮点数或整数。
        - **TypeError** - `value` 不是浮点数或整数。
