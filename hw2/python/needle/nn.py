"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []




class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype)
        self.weight = Parameter(self.weight)
        if bias:
            bias = init.kaiming_uniform(out_features, 1, device=device, dtype=dtype)
            self.bias = ops.reshape(bias, (1, out_features))
            self.bias = Parameter(self.bias)
        else:
            self.bias = None

    def forward(self, X: Tensor) -> Tensor:
        out = X @ self.weight
        if self.bias is not None:
            bias = self.bias.broadcast_to(out.shape)
            out = out + bias
        return out



class Flatten(Module):
    def forward(self, X):
        batch_size = X.shape[0]
        return ops.reshape(X, (batch_size, -1))


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.relu(x)


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        for m in self.modules:
            x = m(x)
        return x


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        log_sum = ops.logsumexp(logits, axes=1)
        y_one_hot = init.one_hot(logits.shape[1], y)
        z_y = ops.summation(logits * y_one_hot, axes=1)
        return ops.summation(log_sum - z_y) / logits.shape[0]



class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(1, dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(1, dim, device=device, dtype=dtype))
        self.running_mean = init.zeros(1, dim, device=device, dtype=dtype)
        self.running_var = init.ones(1, dim, device=device, dtype=dtype)
        ### END YOUR SOLUTION


    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.shape[0]
        broadcast_weight = ops.broadcast_to(self.weight, x.shape)
        broadcast_bias = ops.broadcast_to(self.bias, x.shape)
        if self.training:
            # E[x]
            mean_x_batch = ops.summation(x, axes=0) / batch_size
            mean_x_batch = ops.reshape(mean_x_batch, (1, -1)) # 这里（1，-1）是因为按列求平均了，下面的 LayerNorm1d 是按行求平均
            broadcast_mean_x_batch = ops.broadcast_to(mean_x_batch, x.shape)

            # Var[x]
            var_batch = (x - broadcast_mean_x_batch) ** 2
            var_batch = ops.summation(var_batch, axes=0) / batch_size
            var_batch = ops.reshape(var_batch, (1, -1))
            std_dev_batch = (var_batch + self.eps) ** (0.5)
            broadcast_std_dev_batch = ops.broadcast_to(std_dev_batch, x.shape)

            # Normalize
            x_norm = broadcast_weight * (x - broadcast_mean_x_batch) / broadcast_std_dev_batch + broadcast_bias

            # Update running mean and var
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean_x_batch
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var_batch
        else:
            broadcast_running_mean = ops.broadcast_to(self.running_mean, x.shape)
            broadcast_running_var = ops.broadcast_to(self.running_var, x.shape)
            x_norm = broadcast_weight * (x - broadcast_running_mean) / (broadcast_running_var + self.eps) ** (0.5) + broadcast_bias
        return x_norm



class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = init.ones(1, dim, device=device, dtype=dtype)
        self.bias = init.zeros(1, dim, device=device, dtype=dtype)
        self.weight = Parameter(self.weight)
        self.bias = Parameter(self.bias)

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.shape[0]
        feature_size = x.shape[1]

        # E[x]
        mean_x = ops.summation(x, axes=1) / feature_size
        mean_x = ops.reshape(mean_x, (-1, 1))
        broadcast_meax_x = ops.broadcast_to(mean_x, x.shape)

        # Var[x]
        var = (x - broadcast_meax_x) ** 2
        var = ops.summation(var, axes=1) / feature_size
        var = ops.reshape(var, (-1, 1))
        std_dev_with_eps = (var + self.eps) ** (0.5)
        broadcast_std_dev_with_eps = ops.broadcast_to(std_dev_with_eps, x.shape)

        # Normalize
        x_norm = (x - broadcast_meax_x) / broadcast_std_dev_with_eps

        # Scale and shift
        broadcast_weight = ops.broadcast_to(self.weight, x.shape)
        broadcast_bias = ops.broadcast_to(self.bias, x.shape)
        return x_norm * broadcast_weight + broadcast_bias


class Dropout(Module):
    def __init__(self, p = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            mask = init.randb(*x.shape, p=1 - self.p, device=x.device, dtype=x.dtype) / (1 - self.p)
            return x * mask
        else:
            return x


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return self.fn(x) + x



