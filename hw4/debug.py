import sys
sys.path.append("./tests/hw4")
sys.path.append("./python")

from test_nd_backend import *
from test_cifar_ptb_data import *
from test_conv import *
from needle import backend_ndarray as nd


if __name__ == "__main__":
    """
    Part 1
    """
    # test_stack((5, 5), 0, 2, nd.cpu())
    # test_stack_backward((5, 5), 0, 2, nd.cpu())

    # test_matmul(16, 16, 16, nd.cpu())
    # test_relu((5, 5), nd.cpu())
    # test_tanh_backward((5, 5), nd.cpu())


    """
    Part 2
    """
    # test_cifar10_dataset(True)


    """
    Part 3
    """
    # test_pad_forward({"shape": (10, 32, 32, 8), "padding": ( (0, 0), (2, 2), (2, 2), (0, 0) )}, nd.cpu())
    # test_flip_forward({"shape": (10, 5), "axes": (0,)}, nd.cpu())
    # test_dilate_forward(nd.cpu())
    test_op_conv((3, 16, 16, 8), (3, 3, 8, 16), 1, 2, False, nd.cpu())