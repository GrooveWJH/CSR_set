#include "helper.h"
#include <climits>
#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define NNZ_PER_WG 1024
#define ROWS_PER_WG 64
#define THREADS_PER_WG 64

__global__ void csrScalarReductionKernel(const float *values, const int *cols,
                                         const int *row_delimiters,
                                         const float *x, float *output,
                                         int totalRows) {
  __shared__ float LDS[NNZ_PER_WG];

  int workgroupID = blockIdx.x;
  int localTid = threadIdx.x;
  int startRow = workgroupID * ROWS_PER_WG;
  int stopRow = (workgroupID + 1) * ROWS_PER_WG;

  int globalTid = workgroupID * THREADS_PER_WG + localTid;

  // If the thread is within the bounds of the grid
  if (startRow + localTid < totalRows) {
    int first_col = row_delimiters[startRow];
    int last_col = row_delimiters[startRow + 1];

    for (int i = localTid; i < NNZ_PER_WG; i += THREADS_PER_WG) {
      LDS[i] = values[first_col + i] * x[cols[first_col + i]];
      // if (threadIdx.x == 63 & blockIdx.x == 0)
        printf("B-%d T-%d : LDS[%d] = value[%d](%f) * x[index=%d](%f)= %f \n",
               blockIdx.x, threadIdx.x, i, first_col + i, values[first_col + i],
               cols[first_col + i], x[cols[first_col + i]], LDS[i]);
    }

    __syncthreads();

    float temp = 0;
    // printf("globalTid = %d    ",globalTid);
    for (int i = row_delimiters[globalTid] - first_col;
         i <= row_delimiters[globalTid + 1] - first_col - 1; i++) {
      if (blockIdx.x == 0)
        // printf("-> reduction B%dT%d wanna access
        // LDS[%d]\n",blockIdx.x,threadIdx.x,i);
        temp += LDS[i];
      printf("reduction: B-%d T-%d : += LDS[%d](%f), temp = %f\n", blockIdx.x,
             threadIdx.x, i, LDS[i], temp);
    }

    output[startRow + localTid] = temp;
  }
}

void calculateRowBlocks(int totalRows, const std::vector<int> &row_delimiters,
                        int localSize, std::vector<int> &rowBlocks) {
  int tempSum = 0;
  int lastIdx = 0;
  int ctr = 1;
  rowBlocks.push_back(0); // Start with 0th block

  for (int i = 1; i < totalRows; i++) {
    // Calculate non-zero elements in this row
    tempSum += row_delimiters[i] - row_delimiters[i - 1];

    // Check if the current block can handle this row
    if (tempSum == localSize) {
      rowBlocks.push_back(i);
      ctr++;
      tempSum = 0;
    } else if (tempSum > localSize) {
      if (i - lastIdx > 1) {
        // This extra row doesn't fit into the current block
        rowBlocks.push_back(i - 1);
        ctr++;
        i--; // Step back to the previous row
      } else {
        // If only one row is too large, put it in a separate block
        rowBlocks.push_back(i);
        ctr++;
      }
      tempSum = 0;
    }
    lastIdx = i;
  }
  rowBlocks.push_back(totalRows); // End with the last row
}

int main() {
  std::cout << "start\n";

  // Read the matrix from a file
  std::vector<float> values;
  std::vector<int> cols;
  std::vector<int> row_delimiters;

  std::string matrix_filename =
      "/home/groove/work/microsoft/CSR_set/data/Singular/csr_format.txt";
  readMatrixFromFile(matrix_filename, values, cols, row_delimiters);

  std::string vector_filename =
      "/home/groove/work/microsoft/CSR_set/data/Singular/x.txt";
  std::vector<float> x = readVectorFromFile(vector_filename);

  std::vector<float> output(row_delimiters.size() - 1); // Output vector

  // Allocate memory on the device (GPU)
  float *d_values, *d_x, *d_output;
  int *d_cols, *d_row_delimiters;
  int totalRows = row_delimiters.size() - 1;
  printf("totalrows:%d\n", totalRows);
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

  // Launch CUDA kernel (launching 1 block, THREADS_PER_WG threads per block)
  int numBlocks = (totalRows + ROWS_PER_WG - 1) / ROWS_PER_WG;
  printf("create blocks: %d, threads per block: %d\n", numBlocks,
         THREADS_PER_WG);

  csrScalarReductionKernel<<<numBlocks, THREADS_PER_WG>>>(
      d_values, d_cols, d_row_delimiters, d_x, d_output, totalRows);

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