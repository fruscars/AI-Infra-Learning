//cuda编程
#include <cuda_runtime.h>
#include <stdio.h>

// __global__ 表示这是在 GPU 上运行的核函数
__global__ void matrixMul(int *A, int *B, int *C, int N) {
    // 获取当前线程的行索引和列索引
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        int sum = 0;
        for (int k = 0; k < N; k++) {
            // 计算 C[row][col] 的点积
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main() {
    int N = 1000;
    // 1. 在 GPU 上申请显存
    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N * N * sizeof(int));
    cudaMalloc(&d_B, N * N * sizeof(int));
    cudaMalloc(&d_C, N * N * sizeof(int));

    // 2. 数据从 CPU 拷贝到 GPU (PCIe 总线传输)
    cudaMemcpy(d_A, h_A, N * N * sizeof(int), cudaMemcpyHostToDevice);
    
    // 3. 定义线程块布局 (16x16 的线程块)
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / 16, N / 16);

    // 4. 启动核函数 (启动 16*16*N/16*N/16 个线程同时计算)
    matrixMul<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // 5. 将结果拷贝回 CPU
    cudaMemcpy(h_C, d_C, N * N * sizeof(int), cudaMemcpyDeviceToHost);
    
    return 0;
}