"""The module.
"""
import math
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module


class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        self.weight = Parameter(init.kaiming_uniform(
                in_channels * kernel_size * kernel_size, out_channels, shape=(kernel_size, kernel_size, in_channels, out_channels), dtype=dtype, device=device, requires_grad=True))
        if bias:
            # bound = 1.0 / math.sqrt(in_channels * (kernel_size ** 2))
            self.bias = Parameter(init.rand(out_channels, dtype=dtype, device=device, requires_grad=True))
        else:
            self.bias = None

        self.padding = (kernel_size - 1) // 2

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (N, C, H, W)
        """
        # Transform x from NCHW to NHWC
        x = x.transpose((1, 2)).transpose((2, 3))

        out = ops.conv(x, self.weight, stride=self.stride, padding=self.padding)
        if self.bias is not None:
            bias_broadcast = ops.broadcast_to(self.bias, out.shape)
            out = out + bias_broadcast

        # Transform out from NHWC to NCHW
        out = out.transpose((3, 1)).transpose((3, 2))

        return out