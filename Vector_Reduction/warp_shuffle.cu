#include <iostream>
#include <vector>
#include <iomanip>
#include <cuda_runtime.h>

// 错误检查宏
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// 全局常量：掩码，用于控制 Warp 内部哪些线程参与 Shuffle（0xffffffff 代表全员 32 个线程）
const unsigned int FULL_MASK = 0xffffffff;

// ====================================================================
// Shuffle 优化版核函数：网格跨步循环 + 寄存器级洗牌规约
// ====================================================================
__global__ void vector_reduction_shuffle(const int *d_input, int *d_output, size_t n, int loop_factor) {
    // 依然保留一个极小的共享内存，仅用于 8 个 Warp 之间的最后收尾（只需 8 个槽位，绝对无冲突）
    __shared__ int warp_sums[8]; 

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    // 1. 终极粗粒化：网格跨步循环 (Grid-Stride Loop)
    // 线程块并排大步向前，总共会吃掉「n * loop_factor」规模的数据流！
    int local_sum = 0;
    size_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    // 让 GPU 像传送带一样，把 20 亿级的数据滚滚拉入寄存器
    for (int loop = 0; loop < loop_factor; ++loop) {
        size_t cur_idx = global_idx + loop * stride;
        if (cur_idx < n) {
            local_sum += d_input[cur_idx];
        }
    }

    // 2. 核心大招：利用汇编级 Shuffle 指令在 Warp 内部做树状折叠
    // 数据直接在 32 个寄存器之间并行对折，完全不走 Shared Memory，0 延迟，0 冲突！
    local_sum += __shfl_down_sync(FULL_MASK, local_sum, 16);
    local_sum += __shfl_down_sync(FULL_MASK, local_sum, 8);
    local_sum += __shfl_down_sync(FULL_MASK, local_sum, 4);
    local_sum += __shfl_down_sync(FULL_MASK, local_sum, 2);
    local_sum += __shfl_down_sync(FULL_MASK, local_sum, 1);

    // 3. 跨 Warp 收尾
    // 此时每个 Warp 的 0 号线程（lane_id == 0）手里抓着这个 Warp 的总和
    if (lane_id == 0) {
        warp_sums[warp_id] = local_sum;
    }
    __syncthreads(); // 确保 8 个 Warp 的值都安全放入临时中转站

    // 由第一个 Warp 的前 8 个线程对这 8 个 Warp 的结果做最后的终局洗牌
    if (tid < 8) {
        int final_sum = warp_sums[tid];
        // 8 个线程折叠 3 次即可完成
        final_sum += __shfl_down_sync(FULL_MASK, final_sum, 4);
        final_sum += __shfl_down_sync(FULL_MASK, final_sum, 2);
        final_sum += __shfl_down_sync(FULL_MASK, final_sum, 1);

        // 最终由当前 Block 的 0 号线程写回全局内存
        if (tid == 0) {
            d_output[blockIdx.x] = final_sum;
        }
    }
}

int main() {
    // 我们在显存里只分配 1 亿个物理数据（约 381 MB）
    const size_t N = 100000000; 
    const size_t bytes = N * sizeof(int);

    // 设置循环放大系数：20 倍。也就是说 GPU 会在物理数据上反复迭代
    // 实际处理的虚拟吞吐量 = 1 亿 * 20 = 20 亿级数据量！
    const int loop_factor = 20; 
    const size_t virtual_N = N * loop_factor;
    const size_t virtual_bytes = virtual_N * sizeof(int);

    std::vector<int> h_input(N);
    for (size_t i = 0; i < N; ++i) {
        h_input[i] = static_cast<int>(i % 10);
    }

    std::cout << "==========================================" << std::endl;
    std::cout << "    GPU (CUDA) Warp Shuffle 终极版基准测试  " << std::endl;
    std::cout << "==========================================" << std::endl;
    std::cout << "物理显存占用: " << bytes / (1024 * 1024) << " MB" << std::endl;
    std::cout << "实际测算等效数据量: " << virtual_N << " (" << virtual_bytes / (1024.0 * 1024.0 * 1024.0) << " GB)" << std::endl;

    const int threads_per_block = 256;
    // 网格大小固定为 1024 个 Block，配合跨步循环能吃下无限大的数据
    const int num_blocks = 1024; 

    int *d_input = nullptr;
    int *d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, num_blocks * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // 1. 预热
    vector_reduction_shuffle<<<num_blocks, threads_per_block>>>(d_input, d_output, N, loop_factor);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 2. 正式测量
    CUDA_CHECK(cudaEventRecord(start));
    vector_reduction_shuffle<<<num_blocks, threads_per_block>>>(d_input, d_output, N, loop_factor);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    std::vector<int> h_block_outputs(num_blocks);
    CUDA_CHECK(cudaMemcpy(h_block_outputs.data(), d_output, num_blocks * sizeof(int), cudaMemcpyDeviceToHost));

    long long total_sum = 0;
    for (int val : h_block_outputs) total_sum += val;

    // 计算吞吐有效带宽：使用 20 亿对应的总 GB 数来进行评测
    double gigabytes = static_cast<double>(virtual_bytes) / (1024.0 * 1024.0 * 1024.0);
    double seconds = milliseconds / 1000.0;
    double bandwidth = gigabytes / seconds;

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