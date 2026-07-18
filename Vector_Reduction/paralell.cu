#include <iostream>
#include <vector>
#include <iomanip>
#include <cuda_runtime.h>

// 1. 补上遗漏的错误检查宏
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// ==========================================
// 版本 A：每个线程处理【连续地址】（有冲突、非合并访存版）
// ==========================================
__global__ void vector_reduction_contiguous(const int *d_input, int *d_output, int n, int k) {
    __shared__ int sdata[256];
    __shared__ int warp_sums[8]; 

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 1. 粗粒化阶段：每个线程串行累加其对应的 k 个【连续】元素
    int local_sum = 0;
    int start_idx = idx * k; // 导致跨线程步长变大，全局内存无法合并访问
    for (int i = 0; i < k; ++i) {
        int cur_idx = start_idx + i;
        if (cur_idx < n) {
            local_sum += d_input[cur_idx];
        }
    }
    
    // 写入共享内存
    sdata[tid] = local_sum;
    __syncthreads(); 

    // 2. 阶段二：Warp 内部规约
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    for (int s = 16; s > 0; s >>= 1) {
        if (lane_id < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncwarp(); 
    }

    // 3. 阶段三：跨 Warp 规约
    if (lane_id == 0) {
        warp_sums[warp_id] = sdata[tid]; 
    }
    __syncthreads(); 

    if (tid < 8) {
        int final_sum = warp_sums[tid];
        if (tid < 4) warp_sums[tid] = final_sum = final_sum + warp_sums[tid + 4]; __syncwarp();
        if (tid < 2) warp_sums[tid] = final_sum = final_sum + warp_sums[tid + 2]; __syncwarp();
        if (tid < 1) warp_sums[tid] = final_sum = final_sum + warp_sums[tid + 1]; __syncwarp();

        if (tid == 0) {
            d_output[blockIdx.x] = warp_sums[0];
        }
    }
}

int main() {
    const size_t N = 2000000000; // 20 亿个元素
    const size_t bytes = N * sizeof(int);

    std::vector<int> h_input(N);
    for (size_t i = 0; i < N; ++i) {
        h_input[i] = static_cast<int>(i % 10);
    }

    const int threads_per_block = 256;
    const int k = 8; 
    const int elements_per_block = threads_per_block * k;
    const int num_blocks = (N + elements_per_block - 1) / elements_per_block;

    int *d_input = nullptr;
    int *d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, num_blocks * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // ==========================================
    // 运行测试（这里名称已统一修改为 vector_reduction_contiguous）
    // ==========================================
    // 1. 预热
    vector_reduction_contiguous<<<num_blocks, threads_per_block>>>(d_input, d_output, N, k);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 2. 正式测量
    CUDA_CHECK(cudaEventRecord(start));
    vector_reduction_contiguous<<<num_blocks, threads_per_block>>>(d_input, d_output, N, k);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    std::vector<int> h_block_outputs(num_blocks);
    CUDA_CHECK(cudaMemcpy(h_block_outputs.data(), d_output, num_blocks * sizeof(int), cudaMemcpyDeviceToHost));

    long long total_sum = 0;
    for (int val : h_block_outputs) total_sum += val;

    double gigabytes = static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
    double seconds = milliseconds / 1000.0;
    double bandwidth = gigabytes / seconds;

    std::cout << "==========================================" << std::endl;
    std::cout << "    GPU (CUDA) 连续地址（有冲突）版测试     " << std::endl;
    std::cout << "==========================================" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "计算结果: " << total_sum << std::endl;
    std::cout << "GPU 核函数耗时: " << milliseconds << " ms" << std::endl;
    std::cout << "GPU 有效访存带宽: " << bandwidth << " GB/s" << std::endl;
    std::cout << "==========================================" << std::endl;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    return 0;
}