import sys
sys.path.append('./tests')
from test_autograd_hw import *
# gradient_check(ndl.summation, ndl.Tensor(np.random.randn(5,4)), axes=(1,))
# test_nn_epoch_ndl()
test_matmul_simple_backward()
test_matmul_batched_backward()