#include <vector>
#include <string>


void readMatrixFromFile(const std::string &filename, std::vector<float> &values,
                        std::vector<int> &cols, std::vector<int> &row_delimiters);

std::vector<float> readVectorFromFile(const std::string &filename);