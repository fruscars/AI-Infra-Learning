// matrixMul_naive.cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// 错误检查宏
#define CHECK(call)                                                             \
    {                                                                           \
        const cudaError_t error = call;                                         \
        if (error != cudaSuccess) {                                             \
            fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);              \
            fprintf(stderr, "code: %d, reason: %s\n", error,                    \
                    cudaGetErrorString(error));                                 \
            exit(1);                                                            \
        }                                                                       \
    }

// GPU 核函数：朴素矩阵乘法（不使用共享内存，直接读写全局内存）
__global__ void matrixMulNaive(const int *A, const int *B, int *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        int sum = 0;
        for (int k = 0; k < N; ++k) {
            // 每次循环都要大老远跑去全局内存读取 A 和 B
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main() {
    // ---------- 1. 参数设置 ----------
    const int N = 2048; // 保持 2048x2048，与共享内存版本完全一致
    const size_t bytes = N * N * sizeof(int);

    // ---------- 2. 主机内存分配与快速初始化 ----------
    int *h_A = (int*)malloc(bytes);
    int *h_B = (int*)malloc(bytes);
    int *h_C = (int*)malloc(bytes);

    for (int i = 0; i < N * N; ++i) {
        h_A[i] = 2;
        h_B[i] = 3;
    }

    // ---------- 3. 设备内存分配与数据拷贝 ----------
    int *d_A, *d_B, *d_C;
    CHECK(cudaMalloc(&d_A, bytes));
    CHECK(cudaMalloc(&d_B, bytes));
    CHECK(cudaMalloc(&d_C, bytes));

    CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // ---------- 4. 线程网格配置 ----------
    const int blockSize = 16; 
    dim3 threadsPerBlock(blockSize, blockSize);
    dim3 numBlocks((N + blockSize - 1) / blockSize,
                   (N + blockSize - 1) / blockSize);

    // ---------- 5. 核心吞吐量计时 ----------
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    CHECK(cudaEventRecord(start));
    matrixMulNaive<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CHECK(cudaEventRecord(stop));
    
    CHECK(cudaEventSynchronize(stop)); // 强制同步

    float milliseconds = 0;
    CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    // ---------- 6. 结果拷回 ----------
    CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // ---------- 7. 性能数据打印 ----------
    double ops = 2.0 * (double)N * (double)N * (double)N;
    double gflops = (ops / (milliseconds / 1000.0)) / 1e9;

    printf("================ Naive 性能测试结果 ================\n");
    printf("矩阵大小    : %d x %d\n", N, N);
    printf("网格尺寸    : %d x %d (Block: %d x %d)\n", numBlocks.x, numBlocks.y, blockSize, blockSize);
    printf("内核耗时    : %f ms\n", milliseconds);
    printf("计算吞吐量  : %f GFLOPS\n", gflops);
    printf("====================================================\n");

    // ---------- 8. 资源释放 ----------
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}