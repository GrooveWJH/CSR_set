#include <cusparse.h>
#include <iostream>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include "helper.h"

int main() {
    // Read the matrix in CSR format and the input vector using helper functions
    std::vector<float> values;
    std::vector<int> cols;
    std::vector<int> row_delimiters;
    
    std::string matrix_filename = "/home/groove/work/microsoft/CSR_set/data/csr_format.txt";
    readMatrixFromFile(matrix_filename, values, cols, row_delimiters);
    
    std::string vector_filename = "/home/groove/work/microsoft/CSR_set/data/x.txt";
    std::vector<float> x = readVectorFromFile(vector_filename);
    
    // Determine matrix dimensions and non-zero count
    int M = row_delimiters.size() - 1;  // number of rows
    int nnz = values.size();            // number of non-zero elements
    int N = x.size();                   // number of columns (assumed from x)
    
    // Allocate the output vector on the host
    std::vector<float> output(M, 0.0f);
    
    // Allocate device memory for CSR components and the input/output vectors
    float *d_values = nullptr, *d_x = nullptr, *d_y = nullptr;
    int *d_cols = nullptr, *d_row_delimiters = nullptr;
    cudaMalloc((void**)&d_values, nnz * sizeof(float));
    cudaMalloc((void**)&d_cols, nnz * sizeof(int));
    cudaMalloc((void**)&d_row_delimiters, (M + 1) * sizeof(int));
    cudaMalloc((void**)&d_x, N * sizeof(float));
    cudaMalloc((void**)&d_y, M * sizeof(float));
    
    // Copy the CSR matrix and vector x from host to device
    cudaMemcpy(d_values, values.data(), nnz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cols, cols.data(), nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_delimiters, row_delimiters.data(), (M + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Create cuSPARSE handle and descriptors for CSR matrix and dense vectors
    cusparseHandle_t handle = nullptr;
    cusparseSpMatDescr_t matA = nullptr;
    cusparseDnVecDescr_t vecX = nullptr, vecY = nullptr;
    
    cusparseCreate(&handle);
    
    // Create CSR matrix descriptor (using zero-based indexing)
    cusparseCreateCsr(&matA, M, N, nnz,
                        d_row_delimiters, d_cols, d_values,
                        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    
    // Create dense vector descriptors for input x and output y
    cusparseCreateDnVec(&vecX, N, d_x, CUDA_R_32F);
    cusparseCreateDnVec(&vecY, M, d_y, CUDA_R_32F);
    
    // Setup parameters for the SpMV operation: y = alpha * A * x + beta * y
    float alpha = 1.0f;
    float beta = 0.0f;
    size_t bufferSize = 0;
    void* dBuffer = nullptr;
    
    // Query the required buffer size for the SpMV operation
    cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha, matA, vecX, &beta, vecY,
                            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);
    
    if (bufferSize > 0) {
        cudaMalloc(&dBuffer, bufferSize);
    }
    
    double adaptive_time = measureExecutionTime([&]() {
    // Execute the sparse matrix-vector multiplication
    cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                 &alpha, matA, vecX, &beta, vecY,
                 CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer);
    });


    // Copy the result vector from device to host
    cudaMemcpy(output.data(), d_y, M * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Display the result
    std::cout << "Result of SpMV (y = A * x):" << std::endl;
    for (int i = 0; i < M; i++) {
        std::cout << output[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "cuSPARSE execution time: " << adaptive_time << " ms\n";
    // Release cuSPARSE and CUDA resources
    if (dBuffer) {
        cudaFree(dBuffer);
    }
    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);
    cusparseDestroy(handle);
    
    cudaFree(d_values);
    cudaFree(d_cols);
    cudaFree(d_row_delimiters);
    cudaFree(d_x);
    cudaFree(d_y);
    
    return 0;
}
