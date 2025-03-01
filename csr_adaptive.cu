#include "helper.h"
#include <climits>
#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define NNZ_PER_WG 1024
#define ROWS_PER_WG 64    // 默认每个工作组的行数（用于CPU初步划分）
#define THREADS_PER_WG 64 // 默认每个工作组的线程数
#define SMALL_VALUE 32    // 如果一个工作组的行数 <= 32，则采用CSR-Vector，否则采用CSR-Stream

// CPU函数：计算工作组（row blocks）划分
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

// CSR-Vector内核（用于小行块），仅处理groupID指定的行块
__global__ void csrVectorKernel_group(const float *values, const int *cols, 
                                const int *row_delimiters, const float *x, 
                                float *output, int totalRows, 
                                const int *rowBlocks, int groupID) {
    __shared__ float LDS[NNZ_PER_WG];

    int localTid = threadIdx.x;   // 工作组内线程编号
    int startRow = rowBlocks[groupID];         // 当前工作组起始行
    int nextStartRow = rowBlocks[groupID + 1]; // 下一工作组起始行
    int numRows = nextStartRow - startRow;         // 当前工作组内的行数

    int numNonZeroes = row_delimiters[nextStartRow] - row_delimiters[startRow];

    // 加载数据到共享内存（LDS），步长为THREADS_PER_WG
    for (int i = localTid; i < numNonZeroes; i += THREADS_PER_WG) {
        int idx = row_delimiters[startRow] + i;
        LDS[i] = values[idx] * x[cols[idx]];
    }

    __syncthreads(); // 同步确保LDS加载完成

    // 每个线程对分配给它的行做归约
    float sum = 0.0f;
    // 注意：这里我们使用有效线程数：effectiveThreads = min(numRows, THREADS_PER_WG)
    for (int r = localTid; r < numRows; r += THREADS_PER_WG) {
        int base = row_delimiters[startRow];
        int localStart = row_delimiters[startRow + r] - base;
        int localEnd   = row_delimiters[startRow + r + 1] - base;
        float temp = 0.0f;
        for (int j = localStart; j < localEnd; j++) {
            temp += LDS[j];
        }
        // 将每行的结果写到输出中
        output[startRow + r] = temp;
    }
}

// CSR-Stream内核（用于大行块），仅处理groupID指定的行块
__global__ void csrStreamKernel_group(const float *values, const int *cols, 
                                const int *row_delimiters, const float *x, 
                                float *output, int totalRows, 
                                const int *rowBlocks, int groupID) {
    __shared__ float LDS[NNZ_PER_WG];

    int localTid = threadIdx.x;   // 工作组内线程编号
    int startRow = rowBlocks[groupID];         // 当前工作组起始行
    int nextStartRow = rowBlocks[groupID + 1]; // 下一工作组起始行
    int numRows = nextStartRow - startRow;         // 当前工作组内行数

    int numNonZeroes = row_delimiters[nextStartRow] - row_delimiters[startRow];

    // 计算有效线程数：如果当前行数少于 THREADS_PER_WG，则只用实际行数
    int effectiveThreads = (numRows < THREADS_PER_WG) ? numRows : THREADS_PER_WG;

    // 按有效线程数跨步加载数据到共享内存
    for (int i = localTid; i < numNonZeroes; i += effectiveThreads) {
        int idx = row_delimiters[startRow] + i;
        LDS[i] = values[idx] * x[cols[idx]];
    }
    __syncthreads(); // 同步确保LDS加载完成

    // 每个有效线程处理自己对应的行，利用LDS内的相对偏移进行归约
    if (localTid < effectiveThreads) {
        int base = row_delimiters[startRow];
        int localStart = row_delimiters[startRow + localTid] - base;
        int localEnd   = row_delimiters[startRow + localTid + 1] - base;
        float temp = 0.0f;
        for (int j = localStart; j < localEnd; j++) {
            // 可选调试输出
            // printf("Group %d, T%d: processing LDS[%d] = %f\n", groupID, localTid, j, LDS[j]);
            temp += LDS[j];
        }
        output[startRow + localTid] = temp;
    }
}

// Host Adaptive function (Adaptive CSR)
// 根据每个工作组的行数决定调用CSR-Stream还是CSR-Vector内核
__host__ void csrAdaptiveHost(const float *d_values, const int *d_cols,
                     const int *d_row_delimiters, const float *d_x,
                     float *d_output, int totalRows, const int *d_rowBlocks,
                     int numGroups) {
    // 对每个工作组分别选择调用对应的内核
    for (int groupID = 0; groupID < numGroups; groupID++) {
        // 从设备中拷贝当前组的起始和下一组的起始行号到主机
        int h_start, h_next;
        cudaMemcpy(&h_start, d_rowBlocks + groupID, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_next, d_rowBlocks + groupID + 1, sizeof(int), cudaMemcpyDeviceToHost);
        int numRows = h_next - h_start;
        if (numRows > SMALL_VALUE) {
            // 使用CSR-Stream方式
            csrStreamKernel_group<<<1, THREADS_PER_WG>>>(d_values, d_cols, d_row_delimiters, d_x, d_output, totalRows, d_rowBlocks, groupID);
        } else {
            // 使用CSR-Vector方式
            csrVectorKernel_group<<<1, THREADS_PER_WG>>>(d_values, d_cols, d_row_delimiters, d_x, d_output, totalRows, d_rowBlocks, groupID);
        }
        cudaDeviceSynchronize();
    }
}

int main() {
    std::cout << "start\n";

    std::vector<float> values;
    std::vector<int> cols;
    std::vector<int> row_delimiters;

    std::string matrix_filename = "/home/groove/work/microsoft/CSR_set/data/csr_format.txt";
    readMatrixFromFile(matrix_filename, values, cols, row_delimiters);
    std::string vector_filename = "/home/groove/work/microsoft/CSR_set/data/x.txt";
    std::vector<float> x = readVectorFromFile(vector_filename);
    std::vector<float> output(row_delimiters.size() - 1);

    // Allocate memory on GPU
    float *d_values, *d_x, *d_output;
    int *d_cols, *d_row_delimiters;
    int *d_rowBlocks;
    int totalRows = row_delimiters.size() - 1;

    // Calculate row blocks (workgroups)
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

    // Copy data from host to device
    cudaMemcpy(d_values, values.data(), values.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_cols, cols.data(), cols.size() * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_delimiters, row_delimiters.data(),
               row_delimiters.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rowBlocks, rowBlocks.data(), rowBlocks.size() * sizeof(int),
               cudaMemcpyHostToDevice);

    // 计算工作组数量
    int numGroups = rowBlocks.size() - 1;
    printf("create %d workgroups, each with up to %d threads\n", numGroups, THREADS_PER_WG);

    // 调用Adaptive Host函数，由Adaptive函数内部根据每个工作组的行数调用对应的CSR内核
    csrAdaptiveHost(d_values, d_cols, d_row_delimiters, d_x, d_output, totalRows, d_rowBlocks, numGroups);

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
    cudaFree(d_rowBlocks);

    std::cout << "done";
    return 0;
}