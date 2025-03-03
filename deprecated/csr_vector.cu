#include "helper.h"
#include <climits>
#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define numBlocks 4

int getMaxNNZperRow(const std::vector<int> &row_delimiters) {
  int maxNNZ = 0;
  for (size_t i = 0; i < row_delimiters.size() - 1; i++) {
    int rowNNZ = row_delimiters[i + 1] - row_delimiters[i];
    maxNNZ = std::max(maxNNZ, rowNNZ);
  }
  return maxNNZ;
}

// CSR-Vector Kernel
__global__ void csrVectorKernel_group(const float *values, const int *cols,
                                      const int *row_delimiters, const float *x,
                                      float *output, int totalRows,
                                      const int *rowBlocks, int groupID) {
  __shared__ float LDS[NNZ_PER_WG];

  int localTid = threadIdx.x;
  int startRow = rowBlocks[groupID];
  int nextStartRow = rowBlocks[groupID + 1];
  int numRows = nextStartRow - startRow;

  int numNonZeroes = row_delimiters[nextStartRow] - row_delimiters[startRow];

  // stride parallelism
  for (int i = localTid; i < numNonZeroes; i += THREADS_PER_WG) {
    int idx = row_delimiters[startRow] + i;
    LDS[i] = values[idx] * x[cols[idx]];
  }

  __syncthreads(); // 同步确保LDS加载完成

  // 每个线程对分配给它的行做归约
  float sum = 0.0f;
  // 注意：这里我们使用有效线程数：effectiveThreads = min(numRows,
  // THREADS_PER_WG)
  for (int r = localTid; r < numRows; r += THREADS_PER_WG) {
    int base = row_delimiters[startRow];
    int localStart = row_delimiters[startRow + r] - base;
    int localEnd = row_delimiters[startRow + r + 1] - base;
    float temp = 0.0f;
    for (int j = localStart; j < localEnd; j++) {
      temp += LDS[j];
    }
    // 将每行的结果写到输出中
    output[startRow + r] = temp;
  }
}

int main() {
  std::cout << "start\n";

  std::vector<float> values;
  std::vector<int> cols;
  std::vector<int> row_delimiters;

  std::string matrix_filename =
      "/home/groove/work/microsoft/CSR_set/data/csr_format.txt";
  readMatrixFromFile(matrix_filename, values, cols, row_delimiters);

  std::string vector_filename =
      "/home/groove/work/microsoft/CSR_set/data/x.txt";
  std::vector<float> x = readVectorFromFile(vector_filename);

  std::vector<float> output(row_delimiters.size() - 1);

  // Allocate memory on GPU
  float *d_values, *d_x, *d_output;
  int *d_cols, *d_row_delimiters;
  int totalRows = row_delimiters.size() - 1;
  int maxNNZperROW = getMaxNNZperRow(row_delimiters);

  cudaMalloc(&d_values, values.size() * sizeof(float));
  cudaMalloc(&d_cols, cols.size() * sizeof(int));
  cudaMalloc(&d_row_delimiters, row_delimiters.size() * sizeof(int));
  cudaMalloc(&d_x, x.size() * sizeof(float));
  cudaMalloc(&d_output, output.size() * sizeof(float));

  // Copy data from host to device
  cudaMemcpy(d_values, values.data(), values.size() * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_cols, cols.data(), cols.size() * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_row_delimiters, row_delimiters.data(),
             row_delimiters.size() * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_x, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice);

  std::cout << "Number of blocks: " << numBlocks << std::endl;

  for (int i = 0; i < numBlocks; ++i) {
    int startRow = i * maxNNZperROW;
    int endRow = std::min(startRow + maxNNZperROW, totalRows);
    std::cout << "Block " << i << ": Start Row = " << startRow
              << ", End Row = " << endRow << "\t";
    std::cout << "Number of threads in this block: "
              << maxNNZperROW * (totalRows / numBlocks) << std::endl;
  }

  int numThread = maxNNZperROW * (totalRows / numBlocks);
  csrVectorKernel<<<numBlocks, numThread>>>(
      d_values, d_cols, d_row_delimiters, d_x, d_output, totalRows, numThread);

  cudaDeviceSynchronize();

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    return -1;
  }

  cudaMemcpy(output.data(), d_output, output.size() * sizeof(float),
             cudaMemcpyDeviceToHost);

  std::cout << "Output: ";
  for (const auto &val : output) {
    std::cout << val << " ";
  }
  std::cout << std::endl;

  cudaFree(d_values);
  cudaFree(d_cols);
  cudaFree(d_row_delimiters);
  cudaFree(d_x);
  cudaFree(d_output);

  std::cout << "done";
  return 0;
}