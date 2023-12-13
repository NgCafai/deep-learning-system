import sys
sys.path.append("./tests/hw3")
sys.path.append("./python")

from test_ndarray import *
from needle import backend_ndarray as nd


if __name__ == "__main__":
    test_getitem(device=nd.cpu(), params={"shape": (8, 8, 2, 2, 2, 2), "fn": lambda X: X[1:3, 5:8, 1:2, 0:1, 0:1, 1:2]})