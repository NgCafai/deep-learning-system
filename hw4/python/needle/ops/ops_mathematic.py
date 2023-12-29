"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
# import numpy
import math

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND
from .ops_tuple import *


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a**b

    def gradient(self, out_grad, node):
        if not isinstance(node.inputs[0], NDArray) or not isinstance(
            node.inputs[1], NDArray
        ):
            raise ValueError("Both inputs must be tensors (NDArray).")

        a, b = node.inputs[0], node.inputs[1]
        grad_a = out_grad * b * (a ** (b - 1))
        grad_b = out_grad * (a**b) * log(a)
        return grad_a, grad_b


def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return array_api.power(a, self.scalar)

    def gradient(self, out_grad, node):
        # node.inputs 是一个 list，不要直接用
        return out_grad * self.scalar * power_scalar(node.inputs[0], self.scalar - 1)


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        return a / b

    def gradient(self, out_grad, node):
        a, b = node.inputs
        return out_grad / b, -out_grad * a / (b * b)


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        return a / self.scalar

    def gradient(self, out_grad, node):
        return out_grad / self.scalar


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        # reverses the order of two axes (axis1, axis2), defaults to the last two axes (1 input, axes - tuple)
        axes = list(range(a.ndim))
        if self.axes is None:
            self.axes = axes[-2:]  # last two axes
        axes[self.axes[0]], axes[self.axes[1]
                                 ] = axes[self.axes[1]], axes[self.axes[0]]

        return array_api.transpose(a, axes)

    def gradient(self, out_grad, node):
        return transpose(out_grad, self.axes)


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.reshape(a, self.shape)

    def gradient(self, out_grad, node):
        input_shape = node.inputs[0].shape
        return reshape(out_grad, input_shape)


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        input_shape = node.inputs[0].shape
        output_shape = out_grad.shape

        # 先生成一个和 output_shape 长度一样的 zero list，然后把 input_shape 的值放到最后
        input_shape_adjusted = [0] * len(output_shape)
        if len(input_shape) > 0:
            input_shape_adjusted[-len(input_shape):] = input_shape[:]

        # 找到 input_shape_adjusted 和 output_shape 不同的位置，这些位置就是需要 reduce 的维度
        axes_to_reduce = [axis for axis, (in_dim, out_dim) in enumerate(
            zip(input_shape_adjusted, output_shape)) if in_dim != out_dim]

        # grad = summation(out_grad, tuple(axes_to_reduce))
        
        # The sum() function in ndarray.py only support sum on one axis at each time
        grad = out_grad
        for axis in axes_to_reduce:
            grad = summation(grad, axis, keepdims=True)

        if len(input_shape) > 0:
            grad = reshape(grad, input_shape)
        return grad


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None, keepdims=False):
        self.axes = axes
        self.keepdims = keepdims

    def compute(self, a):
        return array_api.sum(a, self.axes, keepdims=self.keepdims)

    def gradient(self, out_grad, node):
        reduce_shape = list(node.inputs[0].shape)
        if self.axes is not None:
            if isinstance(self.axes, Number):
                self.axes = (self.axes,)
            for axis in self.axes:
                reduce_shape[axis] = 1
            grad = reshape(out_grad, reduce_shape)
        else:
            grad = out_grad
        return broadcast_to(grad, node.inputs[0].shape)


def summation(a, axes=None, keepdims=False):
    return Summation(axes, keepdims)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        return array_api.matmul(a, b)

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        # Gradient with respect to A is out_grad * B^T
        grad_A = matmul(out_grad, transpose(rhs))
        # Gradient with respect to B is A^T * out_grad
        grad_B = matmul(transpose(lhs), out_grad)

        # Adjust the shape of the gradients if necessary.
        # For example, if lhs.shape is (5, 4), and rhs.shape is (6, 6, 4, 3)
        # then out_grad.shape is (6, 6, 5, 3),
        # and now grad_A.shape is (6, 6, 5, 4) and grad_B.shape is (6, 6, 4, 3).
        # So we need to sum over the first two axes of grad_A.
        if grad_A.shape != lhs.shape:
            grad_A = summation(grad_A, tuple(
                range(len(grad_A.shape) - len(lhs.shape))))
        if grad_B.shape != rhs.shape:
            grad_B = summation(grad_B, tuple(
                range(len(grad_B.shape) - len(rhs.shape))))
        assert (grad_A.shape == lhs.shape)
        assert (grad_B.shape == rhs.shape)

        return grad_A, grad_B


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        return -a

    def gradient(self, out_grad, node):
        return -out_grad


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        return array_api.log(a)

    def gradient(self, out_grad, node):
        return divide(out_grad, node.inputs[0])


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        return array_api.exp(a)

    def gradient(self, out_grad, node):
        inputs = node.inputs[0]
        return out_grad * exp(inputs)


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        zeros = array_api.full(a.shape, 0, dtype=a.dtype, device=a.device)
        res = array_api.maximum(a, zeros)
        return res

    def gradient(self, out_grad, node):
        input_data = node.inputs[0].realize_cached_data()
        # mask = array_api.ones_like(input_data)
        # mask[input_data < 0] = 0
        # return out_grad * Tensor(mask)
        mask = input_data >= 0
        return out_grad * mask
    

def relu(a):
    return ReLU()(a)


class Tanh(TensorOp):
    def compute(self, a: NDArray) -> NDArray:
        return array_api.tanh(a)

    def gradient(self, out_grad, node):
        input_data = node.inputs[0].realize_cached_data()
        return out_grad * (1 - array_api.tanh(input_data) ** 2)


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: Tuple[NDArray]) -> NDArray:
        # inputs: Tuple[NDArray] = args.tuple()
        return array_api.stack(args, self.axis)

    def gradient(self, out_grad, node):
        return Split(self.axis)(out_grad)


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A: NDArray) -> Tuple[NDArray]:
        return tuple(array_api.split(A, self.axis))

    def gradient(self, out_grad, node):
        return Stack(self.axis)(out_grad)


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        return array_api.flip(a, self.axes)

    def gradient(self, out_grad, node):
        return flip(out_grad, self.axes)


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        new_shape = list(a.shape)
        slices = [slice(None)] * len(a.shape)
        for axis in self.axes:
            new_shape[axis] *= (self.dilation + 1)
            slices[axis] = slice(None, None, self.dilation + 1)

        out = array_api.full(new_shape, 0.0, dtype=a.dtype, device=a.device)
        out[tuple(slices)] = a
        return out

    def gradient(self, out_grad, node):
        grad = undilate(out_grad, self.axes, self.dilation)
        return grad


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        slices = [slice(None)] * len(a.shape)
        for axis in self.axes:
            slices[axis] = slice(None, None, self.dilation + 1)

        out = a[tuple(slices)]
        return out

    def gradient(self, out_grad, node):
        grad = dilate(out_grad, self.axes, self.dilation)
        return grad


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A: NDArray, B: NDArray) -> NDArray:
        """
        Computes the convolution of A and B.
        Parameters:
        A - input NDArray in NHWC format
        B - kernel
        """
        pad_axes = [(0, 0)] + [(self.padding, self.padding)] * \
            (A.ndim - 2) + [(0, 0)]
        A = A.pad(pad_axes)

        N, H, W, C_in = A.shape
        K, _, _, C_out = B.shape
        Ns, Hs, Ws, Cs = A.strides

        inner_dim = K * K * C_in
        A = (
            A.as_strided(
                shape=(N, H-K+1, W-K+1, K, K, C_in), strides=(Ns, Hs, Ws, Hs, Ws, Cs))
            .compact()
        )
        A = A.reshape((math.prod(A.shape[:-3]), inner_dim))
        B = B.compact()
        out = A @ B.reshape((inner_dim, C_out))
        out = out.reshape((N, H-K+1, W-K+1, C_out))

        # for stride > 1, we need to subsample the output
        if self.stride > 1:
            slices = [slice(None)] * len(out.shape)
            slices[1] = slice(None, None, self.stride)
            slices[2] = slice(None, None, self.stride)
            out = out[tuple(slices)]
        return out

    def gradient(self, out_grad, node):
        A, B = node.inputs # A and B are of type Tensor
        N, H, W, C_in = A.shape
        K, _, _, C_out = B.shape
        Ns, Hs, Ws, Cs = A.realize_cached_data().strides

        if self.stride > 1:
            out_grad = dilate(out_grad, (1, 2), self.stride - 1)

        """
        gradient of A
        """
        # B_t = B.flip((0, 1)).permute((0, 1, 3, 2))
        B_t = flip(B, (0, 1)).transpose((2, 3))
        A_grad = conv(out_grad, B_t, padding=K-1-self.padding)

        """
        gradient of B(the weight)
        """
        # A_t = A.permute((3, 1, 2, 0))
        A_t = A.transpose((0, 3))
        # out_grad_t = out_grad.permute((1, 2, 0, 3))
        out_grad_t = out_grad.transpose((0, 2)).transpose((0, 1))
        B_grad_t = conv(A_t, out_grad_t, padding=self.padding)
        # B_grad = B_grad_t.permute((1, 2, 0, 3))
        B_grad = B_grad_t.transpose((0, 2)).transpose((0, 1))

        return A_grad, B_grad



def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)
