
// 共享内存矩阵乘法示例代码,这里的共享内存搬运，线程块与矩阵映射的很好，数据搬运即不重复也不缺失，而且数据还会被线程多次使用
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
// 错误检查宏，方便定位 CUDA 调用错误
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
#define TILE_WIDTH 16  // 对应你主函数里的 blockSize = 16

// GPU 核函数：使用共享内存的分块矩阵乘法 C = A * B
__global__ void matrixMulShared(const int *A, const int *B, int *C, int N) {
    // 1. 申请共享内存，大小与 Thread Block 一致 (16x16)
    __shared__ int ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ int ds_B[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;  int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    // 计算当前线程在输出矩阵 C 中对应的全局行和列
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;
    int sum = 0;

    // 2. 阶段性循环：以 TILE_WIDTH 为步长，遍历整行/整列
    for (int m = 0; m < (N + TILE_WIDTH - 1) / TILE_WIDTH; ++m) {
        
        // 3. 协同加载：每个线程负责将一个元素从全局内存搬运到共享内存
        
        // 加载 A 的元素到 ds_A
        if (row < N && (m * TILE_WIDTH + tx) < N) {
            ds_A[ty][tx] = A[row * N + m * TILE_WIDTH + tx];
        } else {
            ds_A[ty][tx] = 0; // 越界部分填充 0，不影响乘加结果
        }

        // 加载 B 的元素到 ds_B
        if (col < N && (m * TILE_WIDTH + ty) < N) {
            ds_B[ty][tx] = B[(m * TILE_WIDTH + ty) * N + col];
        } else {
            ds_B[ty][tx] = 0; // 越界部分填充 0
        }

        // 4. 第一处同步：确保共享内存加载完毕
        __syncthreads();

        // 5. 在共享内存中进行局部的乘加计算
        for (int k = 0; k < TILE_WIDTH; ++k) {
            sum += ds_A[ty][k] * ds_B[k][tx];
        }

        // 6. 第二处同步：确保当前分块计算完毕，才可以进入下一轮加载
        __syncthreads();
    }

    // 7. 将最终累加的结果写回全局内存中的 C 矩阵
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

int main() {
    // ---------- 1. 参数设置 ----------
    const int N = 2048; // 增大矩阵到 2048x2048，以便更充分地让 GPU 满载，测出准确性能
    const size_t bytes = N * N * sizeof(int);

    // ---------- 2. 主机内存分配与快初始化 ----------
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
    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 numBlocks((N + TILE_WIDTH - 1) / TILE_WIDTH,
                   (N + TILE_WIDTH - 1) / TILE_WIDTH);

    // ---------- 5. 纯粹的核心吞吐量计时 ----------
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    CHECK(cudaEventRecord(start));
    matrixMulShared<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CHECK(cudaEventRecord(stop));
    
    CHECK(cudaEventSynchronize(stop)); // 强制同步，确保内核执行完毕

    float milliseconds = 0;
    CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    // ---------- 6. 结果拷回（可选，但不影响计时结果） ----------
    CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // ---------- 7. 性能数据打印 ----------
    double ops = 2.0 * (double)N * (double)N * (double)N;
    double gflops = (ops / (milliseconds / 1000.0)) / 1e9;

    printf("================ 性能测试结果 ================\n");
    printf("矩阵大小    : %d x %d\n", N, N);
    printf("网格尺寸    : %d x %d (Block: %d x %d)\n", numBlocks.x, numBlocks.y, TILE_WIDTH, TILE_WIDTH);
    printf("内核耗时    : %f ms\n", milliseconds);
    printf("计算吞吐量  : %f GFLOPS\n", gflops);
    printf("==============================================\n");

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