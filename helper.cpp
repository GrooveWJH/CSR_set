#include "helper.h"
#include <iostream>
#include <fstream>
#include <sstream>

void readMatrixFromFile(const std::string &filename, std::vector<float> &values,
                        std::vector<int> &cols, std::vector<int> &row_delimiters) {
  std::ifstream checkFile(filename);
  if (!checkFile.good()) {
    std::cerr << "File does not exist: " << filename << std::endl;
    return;
  }
  checkFile.close();

  std::ifstream file(filename);
  std::string line;

  if (!file.is_open()) {
    std::cerr << "Error opening file: " << filename << std::endl;
    return;
  }

  int lineCount = 0;

  // Read first line - values
  if (std::getline(file, line)) {
    std::stringstream ss(line);
    float value;
    while (ss >> value) {
      values.push_back(value);
    }
  } else {
    std::cerr << "Error reading values from file: " << filename << std::endl;
    return;
  }

  // Read second line - column indices
  if (std::getline(file, line)) {
    std::stringstream ss(line);
    int colIndex;
    while (ss >> colIndex) {
      cols.push_back(colIndex);
    }
  } else {
    std::cerr << "Error reading column indices from file: " << filename << std::endl;
    return;
  }

  // Read third line - row delimiters
  if (std::getline(file, line)) {
    std::stringstream ss(line);
    int rowPtr;
    while (ss >> rowPtr) {
      row_delimiters.push_back(rowPtr);
    }
  } else {
    std::cerr << "Error reading row delimiters from file: " << filename << std::endl;
    return;
  }
}

std::vector<float> readVectorFromFile(const std::string &filename) {
    std::vector<float> vector;
    
    std::ifstream checkFile(filename);
    if (!checkFile.good()) {
        std::cerr << "File does not exist: " << filename << std::endl;
        return vector;
    }
    checkFile.close();

    std::ifstream file(filename);
    std::string line;
    float value;

    if (file.is_open()) {
        if (std::getline(file, line)) {
            std::istringstream iss(line);
            while (iss >> value) {
                vector.push_back(value);
            }
        } else {
            std::cerr << "Error reading vector from file: " << filename << std::endl;
        }
        file.close();
    } else {
        std::cerr << "Error opening file: " << filename << std::endl;
    }

    return vector;
}

// int main() {
//   std::cout << "start\n";

//   // Read the matrix from a file
//   std::vector<float> values;
//   std::vector<int> cols;
//   std::vector<int> row_delimiters;

//   std::string matrix_filename = "./data/csr_format.txt";
//   readMatrixFromFile(matrix_filename, values, cols, row_delimiters);

//   std::string vector_filename = "./data/x.txt";
//   std::vector<float> x = readVectorFromFile(vector_filename);

//   std::cout << "Row Delimiters Size: " << row_delimiters.size() - 1 << "\n";
//   std::cout << "Vector Size: " << x.size() << "\n";

//   std::cout << "Values: ";
//   for (auto v : values) std::cout << v << " ";
//   std::cout << "\n";

//   std::cout << "Cols: ";
//   for (auto c : cols) std::cout << c << " ";
//   std::cout << "\n";

//   std::cout << "Row Delimiters: ";
//   for (auto r : row_delimiters) std::cout << r << " ";
//   std::cout << "\n";

//   std::cout << "Vector x: ";
//   for (auto v : x) std::cout << v << " ";
//   std::cout << "\n";

//   return 0;
// }