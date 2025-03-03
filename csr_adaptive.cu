#include "helper.h"
#include <climits>
#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>
#include <ostream>
#include <system_error>
#include <vector>

#define NNZ_PER_WG 1024
#define ROWS_PER_WG 64
#define THREADS_PER_WG 64
#define SMALL_VALUE 2

__global__ void computeRowMaxNNZ(const int *row_delimiters, int start, int next,
                                 int *result) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    int maxNNZ = 0;
    for (int i = start; i < next; i++) {
      int nnz = row_delimiters[i + 1] - row_delimiters[i];
      maxNNZ = nnz > maxNNZ ? nnz : maxNNZ;
    }
    *result = maxNNZ;
  }
}

// Calculate the row blocks.
void calculateRowBlocks(int totalRows, const std::vector<int> &row_delimiters,
                        int localSize, std::vector<int> &rowBlocks) {
  int tempSum = 0;
  int lastIdx = 0;
  int ctr = 1;
  rowBlocks.push_back(0); // 从第0行开始

  for (int i = 1; i < totalRows; i++) {
    // 计算当前行的非零元素数目
    tempSum += row_delimiters[i] - row_delimiters[i - 1];

    // 判断当前工作组是否已达到或超过localSize
    if (tempSum == localSize) {
      rowBlocks.push_back(i);
      ctr++;
      tempSum = 0;
    } else if (tempSum > localSize) {
      if (i - lastIdx > 1) {
        // 当前行放不下，则把前一行作为分界点
        rowBlocks.push_back(i - 1);
        ctr++;
        i--; // 回退到上一行重新处理
      } else {
        // 如果只有一行太多，则单独划分一个工作组
        rowBlocks.push_back(i);
        ctr++;
      }
      tempSum = 0;
    }
    lastIdx = i;
  }
  rowBlocks.push_back(totalRows); // 最后一个工作组结束于总行数
}

// CSR-Vector Kernel
__global__ void csrVectorKernel_group_large(const float *values,
                                            const int *cols,
                                            const int *row_delimiters,
                                            const float *x, float *output,
                                            int totalRows, const int *rowBlocks,
                                            int groupID, int maxNNZ) {
  int startRow = rowBlocks[groupID];
  int nextStartRow = rowBlocks[groupID + 1];
  int numRows = nextStartRow - startRow;
  int tid = threadIdx.x;
  int rowInGroup = tid / maxNNZ; // 当前线程对应的组内行号
  int colInRow = tid % maxNNZ;   // 当前线程在该行内的局部索引
  // 如果线程对应的行超出实际行数，直接返回
  if (rowInGroup >= numRows)
    return;
  // 计算全局行号
  int globalRow = startRow + rowInGroup;

  int rowStart = row_delimiters[globalRow];
  int rowEnd = row_delimiters[globalRow + 1];
  int nnz = rowEnd - rowStart;
  // 如果当前线程的列索引在有效范围内，则计算乘积并原子加到 output[globalRow]
  if (colInRow < nnz) {
    float prod = values[rowStart + colInRow] * x[cols[rowStart + colInRow]];
    atomicAdd(&output[globalRow], prod);
  }
}
// CSR-Vector Kernel
__global__ void
csrVectorKernel_group_stride(const float *values, const int *cols,
                             const int *row_delimiters, const float *x,
                             float *output, int totalRows, const int *rowBlocks,
                             int groupID) {
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
    // printf("LDS[1025] = %f",LDS[1025]);
    output[startRow + r] = temp;
  }
}

// CSR-Stream Kernel
__global__ void csrStreamKernel_group(const float *values, const int *cols,
                                      const int *row_delimiters, const float *x,
                                      float *output, int totalRows,
                                      const int *rowBlocks, int groupID) {
  __shared__ float LDS[NNZ_PER_WG];

  int localTid = threadIdx.x;
  int startRow = rowBlocks[groupID];
  int nextStartRow = rowBlocks[groupID + 1];
  int numRows = nextStartRow - startRow;

  int numNonZeroes = row_delimiters[nextStartRow] - row_delimiters[startRow];

  int effectiveThreads = (numRows < THREADS_PER_WG) ? numRows : THREADS_PER_WG;

  // Stride
  for (int i = localTid; i < numNonZeroes; i += effectiveThreads) {
    int idx = row_delimiters[startRow] + i;
    LDS[i] = values[idx] * x[cols[idx]];
  }
  __syncthreads();

  // 每个有效线程处理自己对应的行，利用LDS内的相对偏移进行归约
  if (localTid < effectiveThreads) {
    int base = row_delimiters[startRow];
    int localStart = row_delimiters[startRow + localTid] - base;
    int localEnd = row_delimiters[startRow + localTid + 1] - base;
    float temp = 0.0f;
    for (int j = localStart; j < localEnd; j++) {
      // printf("Group %d, T%d: processing LDS[%d] = %f\n", groupID, localTid,
      // j, LDS[j]);
      temp += LDS[j];
    }
    output[startRow + localTid] = temp;
  }
}

// Adaptive-Kernel
__host__ void csrAdaptiveHost(const float *d_values, const int *d_cols,
                              const int *d_row_delimiters, const float *d_x,
                              float *d_output, int totalRows,
                              const int *d_rowBlocks, int numGroups) {
  for (int groupID = 0; groupID < numGroups; groupID++) {

    int h_start, h_next;
    cudaMemcpy(&h_start, d_rowBlocks + groupID, sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_next, d_rowBlocks + groupID + 1, sizeof(int),
               cudaMemcpyDeviceToHost);
    int numRows = h_next - h_start;
    printf("GroupID = %d, numRows = %d\t", groupID, numRows);

    if (numRows > SMALL_VALUE) {
      std::cout << "Stream: HERE WE GO!" << std::endl;
      csrStreamKernel_group<<<1, THREADS_PER_WG>>>(
          d_values, d_cols, d_row_delimiters, d_x, d_output, totalRows,
          d_rowBlocks, groupID);
    } else {
      std::cout << "Vector: HERE WE GO! - ";

      int *d_maxNNZ;
      int maxNNZ;
      cudaMalloc(&d_maxNNZ, sizeof(int));
      computeRowMaxNNZ<<<1, 1>>>(d_row_delimiters, h_start, h_next, d_maxNNZ);
      cudaMemcpy(&maxNNZ, d_maxNNZ, sizeof(int), cudaMemcpyDeviceToHost);
      cudaFree(d_maxNNZ);
      std::cout << "maxNNZ:" << maxNNZ << std::endl;
      if (maxNNZ > 1024) {
        std::cerr
            << "Error: a block can only have 1024 threads at most, applying "
            << numRows * maxNNZ << " threads\n";
        return;
      } else {
        csrVectorKernel_group_large<<<1, numRows * maxNNZ>>>(
            d_values, d_cols, d_row_delimiters, d_x, d_output, totalRows,
            d_rowBlocks, groupID, maxNNZ);
        // csrVectorKernel_group_stride<<<1, THREADS_PER_WG>>>(
        //     d_values, d_cols, d_row_delimiters, d_x, d_output, totalRows,
        //     d_rowBlocks, groupID);
      }
      cudaDeviceSynchronize();
    }
  }
}

int main() {
  std::cout << "start\n======>\n";

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
  int *d_rowBlocks;
  int totalRows = row_delimiters.size() - 1;

  std::vector<int> rowBlocks;
  calculateRowBlocks(totalRows, row_delimiters, NNZ_PER_WG, rowBlocks);
  std::cout << "Row blocks: ";
  for (const auto &block : rowBlocks) {
    std::cout << block << " ";
  }
  std::cout << std::endl;

  cudaMalloc(&d_values, values.size() * sizeof(float));
  cudaMalloc(&d_cols, cols.size() * sizeof(int));
  cudaMalloc(&d_row_delimiters, row_delimiters.size() * sizeof(int));
  cudaMalloc(&d_x, x.size() * sizeof(float));
  cudaMalloc(&d_output, output.size() * sizeof(float));
  cudaMalloc(&d_rowBlocks, rowBlocks.size() * sizeof(int));

  cudaMemcpy(d_values, values.data(), values.size() * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_cols, cols.data(), cols.size() * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_row_delimiters, row_delimiters.data(),
             row_delimiters.size() * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_x, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_rowBlocks, rowBlocks.data(), rowBlocks.size() * sizeof(int),
             cudaMemcpyHostToDevice);

  int numGroups = rowBlocks.size() - 1;
  printf("create %d workgroups\n",
         numGroups);

  // CSR-Adaptive
  double adaptive_time = measureExecutionTime([&]() {
    csrAdaptiveHost(d_values, d_cols, d_row_delimiters, d_x, d_output,
                    totalRows, d_rowBlocks, numGroups);
    cudaDeviceSynchronize();
  });

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    return -1;
  }

  cudaMemcpy(output.data(), d_output, output.size() * sizeof(float),
             cudaMemcpyDeviceToHost);

  std::cout << "\n======>\nOutput: ";
  for (const auto &val : output) {
    std::cout << val << " ";
  }
  std::cout << std::endl;

  std::cout << "csrAdaptiveHost execution time: " << adaptive_time << " ms\n";

  cudaFree(d_values);
  cudaFree(d_cols);
  cudaFree(d_row_delimiters);
  cudaFree(d_x);
  cudaFree(d_output);
  cudaFree(d_rowBlocks);

  std::cout << "done";
  return 0;
}