#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>
#include <vector>

namespace py = pybind11;


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    size_t sample_idx = 0;
    auto Z = std::vector<std::vector<float>>(batch, std::vector<float>(k, 0.0));
    // Run batch 
    while(sample_idx < m) {
        if(sample_idx + batch > m) {
            batch = m - sample_idx;
        }

        // Compute Z = normalize(exp(X * theta))
        // The outer two loops iterate over Z, whose size is batch * k
        for(size_t idx = 0; idx < batch; idx++) {
            float row_sum = 0.0;
            for(size_t j = 0; j < k; j++) {
                Z[idx][j] = 0.0;
                for(size_t inner_idx = 0; inner_idx < n; inner_idx++) {
                    Z[idx][j] += X[(sample_idx + idx) * n + inner_idx] * theta[inner_idx * k + j];
                }
                Z[idx][j] = std::exp(Z[idx][j]);
                row_sum += Z[idx][j];
            }
            for(size_t j = 0; j < k; j++) {
                Z[idx][j] /= row_sum;
            }
        }

        // Let Z = Z - Iy
        for(size_t idx = 0; idx < batch; idx++) {
            Z[idx][y[sample_idx + idx]] -= 1.0;
        }

        // Compute gradient
        // The outer two loops iterate over theta, whose size is n * k
        for(size_t idx = 0; idx < n; idx++) {
            for(size_t j = 0; j < k; j++) {
                float diff = 0.0;
                for(size_t inner_idx = 0; inner_idx < batch; inner_idx++) {
                    // theta_diff = X^T * Z
                    // theta_diff(idx, j) = sum_{inner_idx=1}^{batch} X^T(idx, inner_idx) * Z(inner_idx, j) 
                    //              = sum_{inner_idx=1}^{batch} X(inner_idx, idx) * Z(inner_idx, j)
                    diff += X[(sample_idx + inner_idx) * n + idx] * Z[inner_idx][j];
                }
                theta[idx * k + j] -= lr * diff / batch;
            }
        }
        sample_idx += batch;
    }
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
