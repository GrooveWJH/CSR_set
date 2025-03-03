#include "helper.h"
#include <climits>
#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>
#include <ostream>
#include <vector>

#define NNZ_PER_WG 1024
#define ROWS_PER_WG 64
#define THREADS_PER_WG 64
#define SMALL_VALUE 2

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
__global__ void csrVectorKernel_group(const float *values, const int *cols,
                                        const int *row_delimiters, const float *x,
                                        float *output, int totalRows,
                                        const int *rowBlocks, int groupID) {
  // 取当前工作组处理的行范围
  int startRow = rowBlocks[groupID];
  int endRow   = rowBlocks[groupID + 1];
  int numRows  = endRow - startRow;

  // 计算当前组内所有行的最大非零元素数 (maxNNZ)
  __shared__ int maxNnz;
  if (threadIdx.x == 0) {
    maxNnz = 0;
    for (int i = 0; i < numRows; i++) {
      int nnz = row_delimiters[startRow + i + 1] - row_delimiters[startRow + i];
      if (nnz > maxNnz)
        maxNnz = nnz;
    }
  }
  __syncthreads();

  // 每个线程根据全局的 threadIdx.x 决定其处理的行和该行中的列位置
  int tid = threadIdx.x;
  int row = tid / maxNnz;   // 行索引（相对于当前 workgroup 内）
  int col = tid % maxNnz;   // 在该行内的局部列索引

  // 如果当前线程分配的行号超过本组行数，则直接返回
  if (row >= numRows) return;

  // 计算当前行在 CSR 中的起始位置和非零数量
  int rowStart = row_delimiters[startRow + row];
  int rowEnd   = row_delimiters[startRow + row + 1];
  int nnz      = rowEnd - rowStart;

  // 每个线程仅处理本行内一个位置的乘积（如果存在）
  float prod = 0.0f;
  if (col < nnz) {
    int idx = rowStart + col;
    prod = values[idx] * x[cols[idx]];
  }

  // 使用共享内存存储每个线程的乘积，尺寸需为 numRows * maxNnz
  extern __shared__ float shmem[]; // 动态分配共享内存
  int index = row * maxNnz + col;
  shmem[index] = prod;
  __syncthreads();

  // 对每一行，利用该行内的所有线程（仅让 col==0 的线程执行归约）做求和
  if (col == 0) {
    float sum = 0.0f;
    for (int j = 0; j < maxNnz; j++) {
      sum += shmem[row * maxNnz + j];
    }
    // 输出对应行的结果
    output[startRow + row] = sum;
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
      std::cout << "Vector: HERE WE GO!" << std::endl;
      csrVectorKernel_group<<<1, THREADS_PER_WG>>>(
          d_values, d_cols, d_row_delimiters, d_x, d_output, totalRows,
          d_rowBlocks, groupID);
    }
    cudaDeviceSynchronize();
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
  printf("create %d workgroups, at least %d of them with up to %d threads\n",
         numGroups, numGroups - 1, NNZ_PER_WG);

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

  std::cout << "Output: ";
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