#include "./fully_connected.h"
#include <iostream>

// SK: CUDA error handling functions
static void HandleError(cudaError_t err, const char *file, int line);
inline void error_check(cudaError_t err, const char* file, int line);
#define HANDLE_ERROR(err) (HandleError( err, __FILE__, __LINE__ ))
#define CUDA_CHECK(err) do { error_check(err, __FILE__, __LINE__); } while(0)

// SK: Forward Prop Fully Connected Kernel
#define TILE_SIZE 16
void getDims(size_t* thrCnt, size_t* blkRWs, size_t* blkCLs, size_t rows, size_t cols);
__global__ void d_transpose(float* a, float* b, int rowsA, int colsA);
__global__ void d_flip_str_order(float* row_mtx, float* col_mtx, size_t rows, size_t cols, bool isRowMajor);
__global__ void gpu_mat_mul(float *d_A, float *d_B, float *d_C, int m, int n, int k);

// Return the result of row * width + col
__host__ __device__ size_t indexify_RM(size_t row, size_t width, size_t col) {
  return ((row * width) + col);
}

// Return the result of col * height + row
__host__ __device__ size_t indexify_CM(size_t row, size_t height, size_t col) {
  return ((col * height) + row);
}

void getThrCnt(size_t* thrCnt) {
  HANDLE_ERROR(cudaFree(0));
  int dev = 0;
  HANDLE_ERROR(cudaGetDevice(&dev));
  HANDLE_ERROR(cudaSetDevice(dev));
  // Find maximum threads per block dimension and use that
  cudaDeviceProp prop;
  HANDLE_ERROR(cudaGetDeviceProperties(&prop, dev));
  
  (*thrCnt) = (int)sqrt((double)prop.maxThreadsDim[0]);
}

void FullyConnected::init() {
  weight.resize(dim_in, dim_out);
  bias.resize(dim_out);
  grad_weight.resize(dim_in, dim_out);
  grad_bias.resize(dim_out);
  set_normal_random(weight.data(), weight.size(), 0, 0.01);
  set_normal_random(bias.data(), bias.size(), 0, 0.01);
}

// SK: Original CPU implementation (without timing)
// void FullyConnected::forward(const Matrix& bottom) {
//   // z = w' * x + b
//   const int n_sample = bottom.cols();
//   top.resize(dim_out, n_sample);
//   top = weight.transpose() * bottom;
//   top.colwise() += bias;
// }

// SK: Modified Parallel implementation
void FullyConnected::forward(const Matrix& bottom) {
  // z = w' * x + b
  // Resize output matrix so weight * bottom is possible
  const int n_sample = bottom.cols();
  top.resize(dim_out, n_sample); // output matrix will be ROWS[dim_out] x COLS[n_sample]

  size_t sz_weight = weight.size() * sizeof(float);
  size_t sz_bottom = bottom.size() * sizeof(float);
  size_t sz_top = top.size() * sizeof(float);

  float *d_weight, *d_bottom, *d_top, *h_top, *h_wght;
  h_top = (float*)malloc(sz_top);
  h_wght = (float*)malloc(sz_weight);
  
  HANDLE_ERROR(cudaMalloc((void**)&d_weight, sz_weight));
  HANDLE_ERROR(cudaMalloc((void**)&d_bottom, sz_bottom));
  HANDLE_ERROR(cudaMalloc((void**)&d_top, sz_top));

  HANDLE_ERROR(cudaMemcpy(d_weight, weight.data(), sz_weight, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(d_bottom, bottom.data(), sz_bottom, cudaMemcpyHostToDevice));

  size_t threadCnt = 0;
  getThrCnt(&threadCnt);
  float num_threads = (float)threadCnt;

  int dimGridSizeY_C = ceil((float)top.rows()/num_threads);
	int dimGridSizeX_C = ceil((float)top.cols()/num_threads);
  dim3 DimGrid(dimGridSizeX_C, dimGridSizeY_C, 1);
  dim3 DimBlock(num_threads, num_threads, 1);
  
  gpu_mat_mul<<<DimGrid, DimBlock>>>(d_weight, d_bottom, d_top, dim_out, dim_in, n_sample);

  HANDLE_ERROR(cudaMemcpy(h_top, d_top, sz_top, cudaMemcpyDeviceToHost));

  top = Eigen::Map<Vector>(h_top, dim_out * n_sample);
  top.resize(dim_out, n_sample);

  top.colwise() += bias;  // for each column in top, add vector bias

  free(h_top);
  HANDLE_ERROR(cudaFree(d_weight));
  HANDLE_ERROR(cudaFree(d_bottom));
  HANDLE_ERROR(cudaFree(d_top));
  CUDA_CHECK(cudaGetLastError());
}

// =======================================================
//                   Device Operations
// =======================================================

// Transpose Matrix A to matrix B
__global__ void d_transpose(float* a, float* b, int rowsA, int colsA) {
   int rIDX = threadIdx.x + blockIdx.x * blockDim.x;
   int cIDX = threadIdx.y + blockIdx.y * blockDim.y;

  // Transpose
  // ensure tIdxs are within the weight matrix dimensions
  if ( rIDX < rowsA && cIDX < colsA) {
    int m_idx = cIDX * rowsA + rIDX;
    int t_idx = rIDX * colsA + cIDX;
    b[t_idx] = a[m_idx];
  }
}

__global__ void gpu_mat_mul(float *d_A, float *d_B, float *d_C, int m, int n, int k){
	/**
	 * Function responsible for performing GPU matrix multiplication w/o shared memory on d_A and d_B
	 * and storing the result in d_C.
	 */
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int a_idx, b_idx, c_idx;
	if(row < m && col < k){
		int i;
		for(i = 0; i < n; i++){
      // row * width + col
      // col * height + row
      //a_idx = i * m + row;
			b_idx = col * n + i;
			c_idx = col * m + row;
			a_idx = row * n + i;
			// b_idx = i * k + col;
			// c_idx = row * k + col;
			d_C[c_idx] += d_A[a_idx] * d_B[b_idx];
		}
	}
}

// =======================================================
//                 End Device Operations
// =======================================================

void FullyConnected::backward(const Matrix& bottom, const Matrix& grad_top) {
  const int n_sample = bottom.cols();
  // d(L)/d(w') = d(L)/d(z) * x'
  // d(L)/d(b) = \sum{ d(L)/d(z_i) }
  grad_weight = bottom * grad_top.transpose();
  grad_bias = grad_top.rowwise().sum();
  // d(L)/d(x) = w * d(L)/d(z)
  grad_bottom.resize(dim_in, n_sample);
  grad_bottom = weight * grad_top;
}

void FullyConnected::update(Optimizer& opt) {
  Vector::AlignedMapType weight_vec(weight.data(), weight.size());
  Vector::AlignedMapType bias_vec(bias.data(), bias.size());
  Vector::ConstAlignedMapType grad_weight_vec(grad_weight.data(),
                                              grad_weight.size());
  Vector::ConstAlignedMapType grad_bias_vec(grad_bias.data(), grad_bias.size());

  opt.update(weight_vec, grad_weight_vec);
  opt.update(bias_vec, grad_bias_vec);
}

std::vector<float> FullyConnected::get_parameters() const {
  std::vector<float> res(weight.size() + bias.size());
  // Copy the data of weights and bias to a long vector
  std::copy(weight.data(), weight.data() + weight.size(), res.begin());
  std::copy(bias.data(), bias.data() + bias.size(),
            res.begin() + weight.size());
  return res;
}

void FullyConnected::set_parameters(const std::vector<float>& param) {
  if (static_cast<int>(param.size()) != weight.size() + bias.size())
      throw std::invalid_argument("Parameter size does not match");
  std::copy(param.begin(), param.begin() + weight.size(), weight.data());
  std::copy(param.begin() + weight.size(), param.end(), bias.data());
}

std::vector<float> FullyConnected::get_derivatives() const {
  std::vector<float> res(grad_weight.size() + grad_bias.size());
  // Copy the data of weights and bias to a long vector
  std::copy(grad_weight.data(), grad_weight.data() + grad_weight.size(),
            res.begin());
  std::copy(grad_bias.data(), grad_bias.data() + grad_bias.size(),
            res.begin() + grad_weight.size());
  return res;
}

static void HandleError(cudaError_t err, const char *file, int line ) {
	/**
	 * Handle error macro provided by instructor for cuda library calls
	 */
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line );
	}
}

inline void error_check(cudaError_t err, const char* file, int line) {
    if(err != cudaSuccess) {
        ::fprintf(stderr, "CUDA ERROR at %s[%d] : %s\n", file, line, cudaGetErrorString(err));
        abort();
    }
}

void getDims(size_t* thrCnt, size_t* blkRWs, size_t* blkCLs, size_t rows, size_t cols) {
   HANDLE_ERROR(cudaFree(0));
   int dev = 0;
   HANDLE_ERROR(cudaGetDevice(&dev));
   HANDLE_ERROR(cudaSetDevice(dev));
   // Find maximum threads per block dimension and use that
   cudaDeviceProp prop;
   HANDLE_ERROR(cudaGetDeviceProperties(&prop, dev));
   
   (*thrCnt) = (int)sqrt((double)prop.maxThreadsDim[0]);
   (*blkRWs) = ((rows+(*thrCnt)-1) / (*thrCnt));
   (*blkCLs) = ((cols+(*thrCnt)-1) / (*thrCnt));
}