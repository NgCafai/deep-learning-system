#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <sstream>

namespace needle {
namespace cuda {

#define BASE_THREAD_NUM 256

#define TILE 4
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);

struct CudaArray {
  CudaArray(const size_t size) {
    cudaError_t err = cudaMalloc(&ptr, size * ELEM_SIZE);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    this->size = size;
  }
  ~CudaArray() { cudaFree(ptr); }
  size_t ptr_as_int() { return (size_t)ptr; }
  
  scalar_t* ptr;
  size_t size;
};

struct CudaDims {
  dim3 block, grid;
};

CudaDims CudaOneDim(size_t size) {
  /**
   * Utility function to get cuda dimensions for 1D call
   */
  CudaDims dim;
  size_t num_blocks = (size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
  dim.block = dim3(BASE_THREAD_NUM, 1, 1);
  dim.grid = dim3(num_blocks, 1, 1);
  return dim;
}

#define MAX_VEC_SIZE 8
struct CudaVec {
  uint32_t size;
  int32_t data[MAX_VEC_SIZE];
};

CudaVec VecToCuda(const std::vector<int32_t>& x) {
  CudaVec shape;
  if (x.size() > MAX_VEC_SIZE) throw std::runtime_error("Exceeded CUDA supported max dimesions");
  shape.size = x.size();
  for (size_t i = 0; i < x.size(); i++) {
    shape.data[i] = x[i];
  }
  return shape;
}

////////////////////////////////////////////////////////////////////////////////
// Fill call
////////////////////////////////////////////////////////////////////////////////

__global__ void FillKernel(scalar_t* out, scalar_t val, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = val;
}

void Fill(CudaArray* out, scalar_t val) {
  CudaDims dim = CudaOneDim(out->size);
  FillKernel<<<dim.grid, dim.block>>>(out->ptr, val, out->size);
}

////////////////////////////////////////////////////////////////////////////////
// Compact and setitem cals
////////////////////////////////////////////////////////////////////////////////

__device__ void CalcStrides(const CudaVec& shape, CudaVec* strides) {
  size_t stride = 1;
  for (int32_t i = shape.size - 1; i >= 0; i--) {
    strides->data[i] = stride;
    stride *= shape.data[i];
  }
}

// Untility function to convert contiguous index i to memory location from strides
__device__ size_t ContiguoutIdxToIdxWithStride(size_t idx_from, const CudaVec& shape, const CudaVec& strides_from,
                                const CudaVec& strides_to, size_t offset_to) {
  /**
   * Utility function to convert an index in the output array to the corresponding index in the
   * input array.  
   */

  // Calculate the index in each dimension
  CudaVec idx_in_each_dim;
  idx_in_each_dim.size = shape.size;
  idx_in_each_dim.data[0] = idx_from / strides_from.data[0];
  size_t remainder = idx_from % strides_from.data[0];
  for (size_t i = 1; i < idx_in_each_dim.size; i++) {
    idx_in_each_dim.data[i] = (int32_t)(remainder / strides_from.data[i]);
    remainder = remainder % strides_from.data[i];
  }

  // Convert to index in input array
  size_t idx_to = offset_to;
  for (size_t i = 0; i < idx_in_each_dim.size; i++) {
    idx_to += idx_in_each_dim.data[i] * strides_to.data[i];
  }
  return idx_to;
}


__global__ void CompactKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset) {
  /**
   * The CUDA kernel for the compact opeation.  This should effectively map a single entry in the 
   * non-compact input a, to the corresponding item (at location gid) in the compact array out.
   * 
   * Args:
   *   a: CUDA pointer to a array
   *   out: CUDA point to out array
   *   size: size of out array
   *   shape: vector of shapes of a and out arrays (of type CudaVec, for past passing to CUDA kernel)
   *   strides: vector of strides of out array
   *   offset: offset of out array
   */
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gid < size) {
    CudaVec strides_out;
    CalcStrides(shape, &strides_out);

    size_t idx_in = ContiguoutIdxToIdxWithStride(gid, shape, strides_out, strides, offset);
    out[gid] = a[idx_in];
  }
}

void Compact(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  /**
   * Compact an array in memory.  Unlike the C++ version, in CUDA this will primarily call the 
   * relevant CUDA kernel.  In this case, we illustrate how you should set this up (i.e., we give 
   * you the code for this fuction, and also the prototype for the CompactKernel() function).  For
   * the functions after this, however, you'll need to define these kernels as you see fit to 
   * execute the underlying function.
   * 
   * Args:
   *   a: non-compact represntation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   */

  // Nothing needs to be added here
  CudaDims dim = CudaOneDim(out->size);
  CompactKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, VecToCuda(shape),
                                         VecToCuda(strides), offset);
}


__global__ void EwiseSetitemKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                                   CudaVec strides_out, size_t offset_out) {
  size_t idx_in = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx_in < size) {
    CudaVec strides_in;
    CalcStrides(shape, &strides_in);

    size_t idx_out = ContiguoutIdxToIdxWithStride(idx_in, shape, strides_in, strides_out, offset_out);
    out[idx_out] = a[idx_in];
  }

}

void EwiseSetitem(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items in a (non-compact) array using CUDA.  Yyou will most likely want to implement a
   * EwiseSetitemKernel() function, similar to those above, that will do the actual work.
   * 
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being compact)
   */
  CudaDims dim = CudaOneDim(a.size);
  EwiseSetitemKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, a.size, VecToCuda(shape),
                                              VecToCuda(strides), offset);
}


__global__ void ScalarSetitemKernel(scalar_t val, scalar_t* out, size_t size, CudaVec shape,
                                    CudaVec strides_out, size_t offset_out) {
  size_t idx_in = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx_in < size) {
    CudaVec strides_in;
    CalcStrides(shape, &strides_in);

    size_t idx_out = ContiguoutIdxToIdxWithStride(idx_in, shape, strides_in, strides_out, offset_out);
    out[idx_out] = val;
  }

}


void ScalarSetitem(size_t size, scalar_t val, CudaArray* out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items is a (non-compact) array
   * 
   * Args:
   *   size: number of elements to write in out array (note that this will note be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the 
   *         product of items in shape, but covenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */
  CudaDims dim = CudaOneDim(size);
  ScalarSetitemKernel<<<dim.grid, dim.block>>>(val, out->ptr, size, VecToCuda(shape),
                                               VecToCuda(strides), offset);
}

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////

__global__ void EwiseAddKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + b[gid];
}

void EwiseAdd(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * Add together two CUDA array
   */
  CudaDims dim = CudaOneDim(out->size);
  EwiseAddKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarAddKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + val;
}

void ScalarAdd(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Add together a CUDA array and a scalar value.
   */
  CudaDims dim = CudaOneDim(out->size);
  ScalarAddKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

/**
 * In the code the follows, use the above template to create analogous elementise
 * and and scalar operators for the following functions.  See the numpy backend for
 * examples of how they should work.
 *   - EwiseMul, ScalarMul
 *   - EwiseDiv, ScalarDiv
 *   - ScalarPower
 *   - EwiseMaximum, ScalarMaximum
 *   - EwiseEq, ScalarEq
 *   - EwiseGe, ScalarGe
 *   - EwiseLog
 *   - EwiseExp
 *   - EwiseTanh
 *
 * If you implement all these naively, there will be a lot of repeated code, so
 * you are welcome (but not required), to use macros or templates to define these
 * functions (however you want to do so, as long as the functions match the proper)
 * signatures above.
 */


////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////

#define EWISE_PAIR_OP(FUN_NAME, OP_EXPR) \
__global__ void FUN_NAME##Kernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) { \
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x; \
  if (gid < size) out[gid] = OP_EXPR; \
} \
\
void FUN_NAME(const CudaArray& a, const CudaArray& b, CudaArray* out) { \
  CudaDims dim = CudaOneDim(out->size); \
  FUN_NAME##Kernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size); \
}

#define EWISE_UNARY_OP(FUN_NAME, OP_EXPR) \
__global__ void FUN_NAME##Kernel(const scalar_t* a, scalar_t* out, size_t size) { \
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x; \
  if (gid < size) out[gid] = OP_EXPR; \
} \
\
void FUN_NAME(const CudaArray& a, CudaArray* out) { \
  CudaDims dim = CudaOneDim(out->size); \
  FUN_NAME##Kernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size); \
}

#define SCALAR_PAIR_OP(FUN_NAME, OP_EXPR) \
__global__ void FUN_NAME##Kernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) { \
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x; \
  if (gid < size) out[gid] = OP_EXPR; \
} \
\
void FUN_NAME(const CudaArray& a, scalar_t val, CudaArray* out) { \
  CudaDims dim = CudaOneDim(out->size); \
  FUN_NAME##Kernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size); \
}

EWISE_PAIR_OP(EwiseMul, a[gid] * b[gid])
SCALAR_PAIR_OP(ScalarMul, a[gid] * val)
EWISE_PAIR_OP(EwiseDiv, a[gid] / b[gid])
SCALAR_PAIR_OP(ScalarDiv, a[gid] / val)
SCALAR_PAIR_OP(ScalarPower, pow(a[gid], val))

EWISE_PAIR_OP(EwiseMaximum, max(a[gid], b[gid]))
SCALAR_PAIR_OP(ScalarMaximum, max(a[gid], val))
EWISE_PAIR_OP(EwiseEq, a[gid] == b[gid])
SCALAR_PAIR_OP(ScalarEq, a[gid] == val)
EWISE_PAIR_OP(EwiseGe, a[gid] >= b[gid])
SCALAR_PAIR_OP(ScalarGe, a[gid] >= val)

EWISE_UNARY_OP(EwiseLog, log(a[gid]))
EWISE_UNARY_OP(EwiseExp, exp(a[gid]))
EWISE_UNARY_OP(EwiseTanh, tanh(a[gid]))


////////////////////////////////////////////////////////////////////////////////
// Matrix multiplication
////////////////////////////////////////////////////////////////////////////////

#define TILE_WIDTH 8

__global__ void MatmulKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, uint32_t M,
                             uint32_t N, uint32_t P) {
  __shared__ scalar_t a_tile[TILE_WIDTH][TILE_WIDTH];
  __shared__ scalar_t b_tile[TILE_WIDTH][TILE_WIDTH];

  uint32_t bx = blockIdx.x, by = blockIdx.y;
  uint32_t tx = threadIdx.x, ty = threadIdx.y;

  uint32_t row_out = by * blockDim.y + ty;
  uint32_t col_out = bx * blockDim.x + tx;

  uint32_t row_a = row_out, col_a;
  uint32_t row_b, col_b = col_out;

  scalar_t dot_sum = 0;
  uint32_t phase_num = (N + TILE_WIDTH - 1) / TILE_WIDTH;
  for(uint32_t phase_idx = 0; phase_idx < phase_num; phase_idx++) {
    // load data
    col_a = phase_idx * TILE_WIDTH + tx;
    row_b = phase_idx * TILE_WIDTH + ty;
    if (row_a < M && col_a < N) {
      a_tile[ty][tx] = a[row_a * N + col_a];
    } else {
      a_tile[ty][tx] = 0;
    }

    if (row_b < N && col_b < P) {
      b_tile[ty][tx] = b[row_b * P + col_b];
    } else {
      b_tile[ty][tx] = 0;
    }

    __syncthreads();

    // do dot product
    for (uint32_t i = 0; i < TILE_WIDTH; i++){
      dot_sum += a_tile[ty][i] * b_tile[i][tx];
    }
    __syncthreads();
  }
  
  if (row_out < M && col_out < P) {
    out[row_out * P + col_out] = dot_sum;
  }
}


void Matmul(const CudaArray& a, const CudaArray& b, CudaArray* out, uint32_t M, uint32_t N,
            uint32_t P) {
  /**
   * Multiply two (compact) matrices into an output (also comapct) matrix.  You will want to look
   * at the lecture and notes on GPU-based linear algebra to see how to do this.  Since ultimately
   * mugrade is just evaluating correctness, you _can_ implement a version that simply parallelizes
   * over (i,j) entries in the output array.  However, to really get the full benefit of this
   * problem, we would encourage you to use cooperative fetching, shared memory register tiling, 
   * and other ideas covered in the class notes.  Note that unlike the tiled matmul function in
   * the CPU backend, here you should implement a single function that works across all size
   * matrices, whether or not they are a multiple of a tile size.  As with previous CUDA
   * implementations, this function here will largely just set up the kernel call, and you should
   * implement the logic in a separate MatmulKernel() call.
   * 
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: comapct 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   M: rows of a / out
   *   N: columns of a / rows of b
   *   P: columns of b / out
   */

  // Use x dimemsion for columns, y dimension for rows
  dim3 grid_dim((P + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH, 1);
  dim3 block_dim(TILE_WIDTH, TILE_WIDTH, 1);
  MatmulKernel<<<grid_dim, block_dim>>>(a.ptr, b.ptr, out->ptr, M, N, P);
}

////////////////////////////////////////////////////////////////////////////////
// Max and sum reductions
////////////////////////////////////////////////////////////////////////////////

__global__ void ReduceMaxKernel(const scalar_t* a, scalar_t* out, size_t size_out, size_t reduce_size) {
  size_t idx_out = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx_out < size_out) {
    size_t offset_in = idx_out * reduce_size;
    scalar_t max_val = a[offset_in];
    for (size_t i = 1; i < reduce_size; i++) {
      max_val = max(max_val, a[offset_in + i]);
    }
    out[idx_out] = max_val;
  }
}

void ReduceMax(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.  Even though it is inefficient,
   * for simplicity you can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  CudaDims dim = CudaOneDim(out->size);
  ReduceMaxKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, reduce_size);
}

__global__ void ReduceSumKernel(const scalar_t* a, scalar_t* out, size_t size_out, size_t reduce_size) {
  size_t idx_out = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx_out < size_out) {
    size_t offset_in = idx_out * reduce_size;
    scalar_t sum_val = a[offset_in];
    for (size_t i = 1; i < reduce_size; i++) {
      sum_val += a[offset_in + i];
    }
    out[idx_out] = sum_val;
  }
}

void ReduceSum(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking summation over `reduce_size` contiguous blocks.  Again, for simplicity you 
   * can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  CudaDims dim = CudaOneDim(out->size);
  ReduceSumKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, reduce_size);
}

}  // namespace cuda
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_cuda, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cuda;

  m.attr("__device_name__") = "cuda";
  m.attr("__tile_size__") = TILE;

  py::class_<CudaArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def_readonly("size", &CudaArray::size)
      .def("ptr", &CudaArray::ptr_as_int);

  // return numpy array, copying from CPU
  m.def("to_numpy", [](const CudaArray& a, std::vector<size_t> shape, std::vector<size_t> strides,
                       size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });

    // copy memory to host
    scalar_t* host_ptr = (scalar_t*)std::malloc(a.size * ELEM_SIZE);
    if (host_ptr == 0) throw std::bad_alloc();
    cudaError_t err = cudaMemcpy(host_ptr, a.ptr, a.size * ELEM_SIZE, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

    // return numpy array
    py::capsule deallocate_buffer(host_ptr, [](void* p) { free(p); });
    return py::array_t<scalar_t>(shape, numpy_strides, host_ptr + offset, deallocate_buffer);
  });

  // copy numpy array to GPU
  m.def("from_numpy", [](py::array_t<scalar_t> a, CudaArray* out) {
    cudaError_t err =
        cudaMemcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
  });

  m.def("fill", Fill);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);

  m.def("ewise_mul", EwiseMul);
  m.def("scalar_mul", ScalarMul);
  m.def("ewise_div", EwiseDiv);
  m.def("scalar_div", ScalarDiv);
  m.def("scalar_power", ScalarPower);

  m.def("ewise_maximum", EwiseMaximum);
  m.def("scalar_maximum", ScalarMaximum);
  m.def("ewise_eq", EwiseEq);
  m.def("scalar_eq", ScalarEq);
  m.def("ewise_ge", EwiseGe);
  m.def("scalar_ge", ScalarGe);

  m.def("ewise_log", EwiseLog);
  m.def("ewise_exp", EwiseExp);
  m.def("ewise_tanh", EwiseTanh);

  m.def("matmul", Matmul);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
}
