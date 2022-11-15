#include "./relu.h"
#include <iostream>
#include <math.h>


static void HandleError(cudaError_t err, const char *file, int line);
inline void error_check(cudaError_t err, const char* file, int line);
//__global__ void relu(float* mat, float* result);
__global__ void relu(float* mat, int mat_size, int rows, int cols);
#define HANDLE_ERROR(err) (HandleError( err, __FILE__, __LINE__ ))
#define CUDA_CHECK(err) do { error_check(err, __FILE__, __LINE__); } while(0)

float cpu_time2(timespec* start, timespec* end){
	/**
	 * Function responsible for returning the ellapsed time in
	 * milliseconds
	 */
	return ((1e9*end->tv_sec + end->tv_nsec) - (1e9*start->tv_sec + 
	start->tv_nsec))/1e6;
}

// void ReLU::forward(const Matrix& bottom) {

//   int rows = bottom.rows();
//   int cols = bottom.cols();

//   int mat_size = rows * cols;

//   // a = z*(z>0)
//   timespec ts, te;
//   clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
//   top = bottom.cwiseMax(0.0);

//   // End timing the CPU implementation
//   clock_gettime(CLOCK_MONOTONIC_RAW, &te);
//   std::cout << "Matrix Size: " << mat_size << " | elapsed time: " << cpu_time2(&ts, &te) << std::endl;
// }

void ReLU::forward(const Matrix& bottom) {
  // a = z*(z>0)

  //std::cout << "ReLU FORWARD:" << std::endl;
  //std::cout << bottom.rows() << ", " << bottom.cols() << std::endl;

  // get rows and cols
  int rows = bottom.rows();
  int cols = bottom.cols();

  int mat_size = rows * cols;
  int mat_mem_size = mat_size * sizeof(float);
  //std::cout << "Matrix size: " << mat_size << std::endl;

  float *d_mat;
  //float *d_result;

  //std::cout << "Allocating Memory" << std::endl;
  // allocate memory for the device matrix
  HANDLE_ERROR(cudaMalloc((void **)&d_mat, mat_mem_size));
  //HANDLE_ERROR(cudaMalloc((void **)&d_result, mat_mem_size));

  CUDA_CHECK(cudaGetLastError());

  //std::cout << "Copying memory" << std::endl;
  // copy memory to GPU ram
  HANDLE_ERROR(cudaMemcpy(d_mat, bottom.data(), mat_size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaGetLastError());

  // how many threads should I use?
  float num_threadsX = 32.00;
  float num_threadsY = 32.00;

  int dimGridSizeX = ceil((float)cols/num_threadsX);
  int dimGridSizeY = ceil((float)rows/num_threadsY);

  // get dimensions
  dim3 DimGrid(dimGridSizeX, dimGridSizeY, 1);
  dim3 DimBlock(num_threadsX, num_threadsY, 1);

  CUDA_CHECK(cudaGetLastError());

  // run kernel
  //std::cout << "Running Kernel" << std::endl;
  //relu<<<DimGrid, DimBlock>>>(d_mat, d_result);

  // TIMING:
  cudaEvent_t start, stop; //declare a start and stop event
  HANDLE_ERROR(cudaEventCreate(&start)); //create both events
  HANDLE_ERROR(cudaEventCreate(&stop));
  HANDLE_ERROR(cudaEventRecord(start)); //insert the start event into the stream

  //std::cout << "TESTING:\n" << 

  relu<<<DimGrid, DimBlock>>>(d_mat, mat_size, rows, cols);

  // End profiling code:
  HANDLE_ERROR(cudaEventRecord(stop)); //insert the stop event into the stream
  cudaThreadSynchronize();
  float milliseconds = 0; //declare a variable to store runtime
  HANDLE_ERROR(cudaEventElapsedTime(&milliseconds, start, stop)); //get the elapsed
  //std::cout << "Matrix size: " << mat_size << " | elapsed time: " << milliseconds << std::endl;

  //std::cout << "Kernel Finished" << std::endl;

  CUDA_CHECK(cudaGetLastError());

  float * result = (float*) malloc(mat_size * sizeof(float));

  //std::cout << "Copying data back:" << std::endl;
  HANDLE_ERROR(cudaMemcpy(result, d_mat, mat_mem_size, cudaMemcpyDeviceToHost));  //fixme d_result?  // or mat_size?
  //std::cout << "Copied back successfully" << std::endl;

  CUDA_CHECK(cudaGetLastError());

  Matrix output = Eigen::Map<Matrix>(result, rows, cols);
  top = output;

  // Free Cuda Memory
  HANDLE_ERROR(cudaFree(d_mat));
  CUDA_CHECK(cudaGetLastError());
} // END FORWARD

//__global__ void relu(float* mat, float* result){
__global__ void relu(float* mat, int mat_size, int rows, int cols){
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int idx = rows * col + row;

  if (idx < mat_size){
    mat[idx] = mat[idx] * (int)(mat[idx] > 0);
  }
}

void ReLU::backward(const Matrix& bottom, const Matrix& grad_top) {
  // d(L)/d(z_i) = d(L)/d(a_i) * d(a_i)/d(z_i)
  //             = d(L)/d(a_i) * 1*(z_i>0)
  Matrix positive = (bottom.array() > 0.0).cast<float>();
  grad_bottom = grad_top.cwiseProduct(positive);
}

// Handle Error

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