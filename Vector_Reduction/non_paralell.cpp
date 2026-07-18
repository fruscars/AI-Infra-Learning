#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>

int main() {
    // 1. 初始化数据：1亿个 int 元素（约占 400MB 内存，确保超出 CPU 缓存，直达内存）
    const size_t N = 100000000; 
    std::vector<int> data(N);
    
    // 用 0 到 9 的循环数填充，防止数据全为 1 被编译器做奇奇怪怪的优化
    for (size_t i = 0; i < N; ++i) {
        data[i] = static_cast<int>(i % 10);
    }

    std::cout << "==========================================" << std::endl;
    std::cout << "        CPUVector Reduction Benchmark         " << std::endl;
    std::cout << "==========================================" << std::endl;
    std::cout << "element num: " << N << " (" << (N * sizeof(int)) / (1024 * 1024) << " MB)" << std::endl;

    // 2. 预热（Warm-up）：让 CPU 进入状态，将数据加载到缓存中（消除首次冷启动误差）
    long long warm_up_sum = 0;
    for (size_t i = 0; i < N; ++i) {
        warm_up_sum += data[i];
    }

    // 3. 正式性能测试
    auto start_time = std::chrono::high_resolution_clock::now();

    long long total_sum = 0; // 使用 long long 防止求和溢出
    for (size_t i = 0; i < N; ++i) {
        total_sum += data[i];
    }

    auto end_time = std::chrono::high_resolution_clock::now();

    // 4. 计算性能指标
    std::chrono::duration<double, std::milli> elapsed_ms = end_time - start_time;
    
    // 计算处理的数据量 (GB)
    double bytes_processed = static_cast<double>(N * sizeof(int));
    double gigabytes = bytes_processed / (1024.0 * 1024.0 * 1024.0);
    
    // 计算带宽 (GB/s) = 数据量(GB) / 时间(秒)
    double seconds = elapsed_ms.count() / 1000.0;
    double bandwidth = gigabytes / seconds;

    // 5. 显示性能结果
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Result: " << total_sum << std::endl;
    std::cout << "consume_time: " << elapsed_ms.count() << " ms" << std::endl;
    std::cout << "validated_bandwidth: " << bandwidth << " GB/s" << std::endl;
    std::cout << "==========================================" << std::endl;

    return 0;
}