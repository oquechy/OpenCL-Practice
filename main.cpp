#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.h>
#include "cl.hpp"

#include <vector>
#include <fstream>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <cassert>

int main() {
    std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;
    std::vector<cl::Kernel> kernels;


    // create platform
    cl::Platform::get(&platforms);
    platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

    // create context
    cl::Context context(devices);

    // create command queue
    cl::CommandQueue queue(context, devices[0]);

    // load opencl source
    std::ifstream cl_file("convolution.cl");
    std::string cl_string(std::istreambuf_iterator<char>(cl_file), (std::istreambuf_iterator<char>()));
    cl::Program::Sources source(1, std::make_pair(cl_string.c_str(),
                                                  cl_string.length() + 1));

    // create program
    cl::Program program(context, source);

    try {
        // compile opencl source
        size_t const block_size = 16;
        program.build(devices, "-D BLOCK_SIZE=16");

        // create a message to send to kernel
        std::ifstream in("input.txt");
        size_t N, M;
        in >> N >> M;

        size_t const a_size = N * N;
        size_t const b_size = M * M;

        float *a = new float[a_size];
        float *b = new float[b_size];
        float *c = new float[a_size];

        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                size_t idx = i * N + j;
                in >> a[idx];
                c[idx] = 0;
            }
        }
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < M; ++j) {
                size_t idx = i * M + j;
                in >> b[idx];
            }
        }

        // allocate device buffer to hold message
        cl::Buffer dev_a(context, CL_MEM_READ_ONLY, sizeof(float) * a_size);
        cl::Buffer dev_b(context, CL_MEM_READ_ONLY, sizeof(float) * b_size);
        cl::Buffer dev_c(context, CL_MEM_WRITE_ONLY, sizeof(float) * a_size);

        // copy from cpu to gpu
        queue.enqueueWriteBuffer(dev_a, CL_TRUE, 0, sizeof(float) * a_size, a);
        queue.enqueueWriteBuffer(dev_b, CL_TRUE, 0, sizeof(float) * b_size, b);

        // load named kernel from opencl source
        cl::Kernel kernel(program, "convolution");
        cl::KernelFunctor convolution(kernel, queue, cl::NullRange, cl::NDRange(N, N),
                                      cl::NDRange(block_size, block_size));

        convolution(dev_a, dev_b, dev_c, (int) N, (int) M);

        queue.enqueueReadBuffer(dev_c, CL_TRUE, 0, sizeof(int) * a_size, c);

        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                std::cout << c[i * N + j] << ' ';
            }
            std::cout << '\n';
        }

        int const HM = (M - 1) / 2;
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                size_t idx = i * N + j;

                float sum = 0;
                for (int k = -HM; k <= HM; ++k) {
                    for (int l = -HM; l <= HM; ++l) {
                        int row = i + k, col = j + l;
                        if (0 <= row && row < N && 0 <= col && col < N) {
                            assert(0 <= (k + HM) && (k + HM) < M);
                            assert(0 <= (l + HM) && (l + HM) < M);
                            sum += a[row * N + col] * b[(k + HM) * M + (l + HM)];
                        }
                    }
                }

                if (c[idx] != sum)
                    std::cout << "c[" << i << "][" << j << "] == " << c[idx] << " != " << sum << std::endl;
            }
        }
    }
    catch (cl::Error &e) {
        std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
    }

    return 0;
}