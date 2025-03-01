import numpy as np
from scipy.sparse import csr_matrix

def generate_sparse_matrix(fixed_nnz_per_row=True, nnz_per_row=16, rows=128):
    if fixed_nnz_per_row:
        matrix = np.zeros((rows, 32))
        for i in range(rows):
            cols = np.random.choice(32, nnz_per_row, replace=False)
            matrix[i, cols] = np.random.rand(nnz_per_row)
        matrix = csr_matrix(matrix)
    return matrix

def calculate_result(matrix, x):
    result = matrix.dot(x)
    return result

def write_matrix_to_files(matrix):
    dense_matrix = matrix.toarray()
    x = np.arange(1, matrix.shape[1] + 1)
    
    with open('./Singular/csr_format.txt', 'w') as f:
        f.write(' '.join(f"{val:.6f}" for val in matrix.data) + '\n')
        f.write(' '.join(str(col) for col in matrix.indices) + '\n')
        f.write(' '.join(str(ptr) for ptr in matrix.indptr) + '\n')
        f.write('\n')
        for row in dense_matrix:
            f.write(' '.join(f"{val:.6f}" for val in row) + '\n')

    with open('./Singular/x.txt', 'w') as f:
        f.write(' '.join(f"{val:.6f}" for val in x) + '\n')

    with open('./Singular/ans.txt', 'w') as f:
        result = calculate_result(matrix, x)
        f.write(' '.join(f"{val:.6f}" for val in result) + '\n')

    with open('./Singular/summary.txt', 'w') as f:
        f.write('# CSR Format Components\n')
        f.write('Values array:\n')
        f.write(' '.join(f"{val:.6f}" for val in matrix.data) + '\n\n')
        f.write('Column indices array:\n')
        f.write(' '.join(str(col) for col in matrix.indices) + '\n\n')
        f.write('Row delimiters array:\n')
        f.write(' '.join(str(ptr) for ptr in matrix.indptr) + '\n\n')
        
        f.write('# Dense Matrix Format\n')
        for row in dense_matrix:
            f.write(' '.join(f"{val:.6f}" for val in row) + '\n')
        f.write('\n')
        
        f.write('# Input Vector x\n')
        f.write(' '.join(f"{val:.6f}" for val in x) + '\n\n')
        
        f.write('# Matrix-Vector Multiplication Result\n')
        result = calculate_result(matrix, x)
        f.write(' '.join(f"{val:.6f}" for val in result) + '\n')

def main():
    matrix = generate_sparse_matrix(nnz_per_row=16,rows=1)
    result = calculate_result(matrix, np.arange(1, matrix.shape[1] + 1))
    print("Calculation result:")
    print(result)
    write_matrix_to_files(matrix)

if __name__ == "__main__":
    main()
