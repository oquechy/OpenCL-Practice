#include <iostream>
#include <fstream>
#include <cstdlib>

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cerr << "provide N & M" << std::endl;
        return 1;
    }

    std::ofstream out("input.txt");
    size_t N = atoi(argv[1]), M = atoi(argv[2]);
    out << N << " " << M << std::endl;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            out << "1 ";
        }
        out << std::endl;
    }
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < M; ++j) {
            out << "1 ";
        }
        out << std::endl;
    }
}
