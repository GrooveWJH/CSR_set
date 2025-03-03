#include <vector>
#include <string>
#include <chrono>
#include <functional>


// 直接在头文件里完成定义 (并加 inline)
template<typename Func, typename... Args>
inline double measureExecutionTime(Func&& f, Args&&... args) {
    auto start = std::chrono::high_resolution_clock::now();
    std::invoke(std::forward<Func>(f), std::forward<Args>(args)...);
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

void readMatrixFromFile(const std::string &filename, std::vector<float> &values,
                        std::vector<int> &cols, std::vector<int> &row_delimiters);

std::vector<float> readVectorFromFile(const std::string &filename);