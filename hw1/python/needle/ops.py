"""Operator implementations."""

from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks
import numpy as array_api


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


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return array_api.power(a, self.scalar)

    def gradient(self, out_grad, node):
        return out_grad * self.scalar * node.inputs


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
        axes[self.axes[0]], axes[self.axes[1]] = axes[self.axes[1]], axes[self.axes[0]]

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
        axes_to_reduce = [axis for axis, (in_dim, out_dim) in enumerate(zip(input_shape_adjusted, output_shape)) if in_dim != out_dim]
        grad = summation(out_grad, tuple(axes_to_reduce))

        if len(input_shape) > 0:
            grad = reshape(grad, input_shape)
        return grad



def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        return array_api.sum(a, self.axes)

    def gradient(self, out_grad, node):
        reduce_shape = list(node.inputs[0].shape)
        if self.axes is not None:
            if not isinstance(self.axes, tuple):
                self.axes = tuple(self.axes)
            for axis in self.axes:
                reduce_shape[axis] = 1
            grad = reshape(out_grad, reduce_shape)
        else:
            grad = out_grad
        return broadcast_to(grad, node.inputs[0].shape)

        


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        return array_api.matmul(a, b)

    def gradient(self, out_grad, node):
        A, B = node.inputs

        # Gradient with respect to A is out_grad * B^T
        grad_A = out_grad @ B.transpose()

        # Gradient with respect to B is A^T * out_grad
        grad_B = A.transpose() @ out_grad

        lhs, rhs = node.inputs
        grad_A = matmul(out_grad, transpose(rhs))
        grad_B = matmul(transpose(lhs), out_grad)
        if grad_A.shape != lhs.shape:
            grad_A = summation(grad_A, tuple(range(len(grad_A.shape) - len(lhs.shape))))
        if grad_B.shape != rhs.shape:
            grad_B = summation(grad_B, tuple(range(len(grad_B.shape) - len(rhs.shape))))
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
        res = a.copy()
        res[res < 0] = 0
        return res

    def gradient(self, out_grad, node):
        out_grad_data = out_grad.realize_cached_data()
        input_data = node.inputs[0].realize_cached_data()
        out_grad_data[input_data < 0] = 0
        return Tensor(out_grad_data)


def relu(a):
    return ReLU()(a)

