from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        max_Z = array_api.max(Z, self.axes, keepdims=True)
        Z = Z - array_api.broadcast_to(max_Z, Z.shape)
        res = array_api.log(array_api.sum(array_api.exp(Z), self.axes))
        res = res + array_api.reshape(max_Z, res.shape)
        return res

    def gradient(self, out_grad, node):
        input_data = node.inputs[0].realize_cached_data()
        max_input = array_api.max(input_data, self.axes, keepdims=True)
        input_data = input_data - array_api.broadcast_to(max_input, input_data.shape)
        sum_exp_z = array_api.sum(array_api.exp(input_data), self.axes, keepdims=True)
        cur_grad = array_api.exp(input_data) / array_api.broadcast_to(sum_exp_z, input_data.shape)

        if out_grad.shape != cur_grad.shape:
            if out_grad.cached_data.size == cur_grad.size:
                out_grad = reshape(out_grad, cur_grad.shape)
            else:
                # 对 out_grad 进行 reshape（比如从 （3,）变成 （3,1），否则 broadcast 时，结果和预期不一致），然后再进行 broadcast_to
                new_shape = list(cur_grad.shape)
                if self.axes is not None:
                    if isinstance(self.axes, Number):
                        self.axes = (self.axes,)
                    for axis in self.axes:
                        new_shape[axis] = 1
                else:
                    new_shape = [1] * len(new_shape)
                out_grad = reshape(out_grad, new_shape)
                out_grad = broadcast_to(out_grad, cur_grad.shape)
        return out_grad * cur_grad

def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

