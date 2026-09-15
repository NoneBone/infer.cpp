#include "base/base.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>
/*
/usr/local/cuda-12.8/bin/nvcc \
  -std=c++17 -O2 -x cu \
  test/single/bufferPool.cpp \
  -I infer/include \
  -L/usr/lib/x86_64-linux-gnu \
  -lnvidia-ml \
  -o /tmp/test_cuda_pool

/tmp/test_cuda_pool 0

/tmp/test_cuda_pool 1
*/
namespace base
{
    nvtxTimeTable nvTT;
    memoryProfiler mP;
}

struct Block
{
    void *ptr;
    size_t bytes;
    bool busy;
};

class DemoAllocator
{
public:
    explicit DemoAllocator(bool enabled) : enabled_(enabled) {}
    void *allocate(size_t bytes)
    {
        if (enabled_)
        {
            // 按照需求尺寸，选择大容器还是小容器
            auto &list = bytes > 1ULL * 1024 * 1024 ? big_ : small_;
            int selected = -1;
            for (int i = 0; i < static_cast<int>(list.size()); ++i)
            {
                auto &b = list[i];
                bool suitable = !b.busy && b.bytes >= bytes;
                bool close_enough = bytes <= 1ULL * 1024 * 1024 ||
                                    b.bytes - bytes < 1ULL * 1024 * 1024;
                // 容器合适，且与需求尺寸接近，就给配给需求
                if (suitable && close_enough &&
                    (selected < 0 || list[selected].bytes > b.bytes))
                    selected = i;
            }
            // 选定的容器，忙碌位置为一
            if (selected >= 0)
            {
                list[selected].busy = true;
                if (bytes <= 1ULL * 1024 * 1024)
                    no_busy_ -= list[selected].bytes;// 从空闲显存统计量 no_busy_ 中减去当前重新被使用的容器大小
                ++reuse_;
                return list[selected].ptr;
            }
        }
        // 正常的分配，或者没有找到合适的容器时
        void *ptr = nullptr;
        check(cudaMalloc(&ptr, bytes));
        ++malloc_;
        if (enabled_)
        {
            auto &list = bytes > 1ULL * 1024 * 1024 ? big_ : small_;
            list.push_back({ptr, bytes, true});
        }
        return ptr;
    }
    void release(void *ptr)
    {
        if (enabled_)
        {
            for (auto &b : small_)
                if (b.ptr == ptr)
                {
                    b.busy = false;
                    no_busy_ += b.bytes;
                    maybe_cleanup();// 空闲量小于 1GB 都先不清理；大于 1GB 就把非busy的清理掉，统计量写 0
                    return;
                }
            for (auto &b : big_)
                if (b.ptr == ptr)
                {
                    b.busy = false;
                    return;
                }
        }
        check(cudaFree(ptr));
        ++free_;
    }
    ~DemoAllocator()
    {
        for (auto &b : small_)
            check(cudaFree(b.ptr));
        for (auto &b : big_)
            check(cudaFree(b.ptr));
    }
    size_t malloc_count() const { return malloc_; }
    size_t free_count() const { return free_; }
    size_t reuse_count() const { return reuse_; }

private:
    static void check(cudaError_t e)
    {
        if (e != cudaSuccess)
        {
            std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(e));
            std::exit(1);
        }
    }
    void maybe_cleanup()
    {
        if (no_busy_ <= 1024ULL * 1024 * 1024)
            return;
        auto keep_busy = [](std::vector<Block> &list)
        {
            std::vector<Block> kept;
            for (auto &b : list)
            {
                if (b.busy)
                    kept.push_back(b);
                else
                {
                    check(cudaFree(b.ptr));
                }
            }
            list.swap(kept);
        };
        keep_busy(small_);
        no_busy_ = 0;
    }
    bool enabled_;
    std::vector<Block> small_, big_;
    size_t no_busy_ = 0;
    size_t malloc_ = 0, free_ = 0, reuse_ = 0;
};

__global__ void simulate_kernel(float *p, size_t n)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        for (int r = 0; r < 100; ++r)
            p[i] = p[i] * 1.000001f + 0.000001f;
}

void checkCuda(cudaError_t e)
{
    if (e != cudaSuccess)
    {
        std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

void run_one(DemoAllocator &a)
{
    // 模拟不同 token 长度/算子 workspace 导致的不同申请尺寸。
    static const size_t patterns[][6] = {
        {64ULL << 20, 128ULL << 20, 256ULL << 20, 128ULL << 20, 768ULL << 10, 2ULL << 20},
        {70ULL << 20, 120ULL << 20, 255ULL << 20, 130ULL << 20, 900ULL << 10, 3ULL << 20},
        {64ULL << 20, 140ULL << 20, 250ULL << 20, 128ULL << 20, 512ULL << 10, 2ULL << 20},
    };
    const size_t *sizes = patterns[static_cast<int>(a.reuse_count() % 3)];
    std::vector<void *> buffers;
    for (int i = 0; i < 6; ++i)
        buffers.push_back(a.allocate(sizes[i]));
    simulate_kernel<<<(sizes[0] / sizeof(float) + 255) / 256, 256>>>(
        static_cast<float *>(buffers[0]), sizes[0] / sizeof(float));
    for (void *p : buffers)
        a.release(p);
    checkCuda(cudaDeviceSynchronize());
}

int main(int argc, char **argv)
{
    bool pool = argc > 1 && std::string(argv[1]) == "1";
    DemoAllocator allocator(pool);
    for (int i = 0; i < 3; ++i)
        run_one(allocator);
    base::mP.mem_push("before");
    base::nvTT.time_push("alloc_and_infer");
    for (int i = 0; i < 20; ++i)
        run_one(allocator);
    base::nvTT.time_pop("alloc_and_infer");
    base::mP.mem_push("after");
    base::nvTT.print_avg_stats();
    base::mP.print_mem_info();
    std::printf("pool=%s, cudaMalloc=%zu, cudaFree=%zu, reuse=%zu\n",
                pool ? "ON" : "OFF", allocator.malloc_count(), allocator.free_count(), allocator.reuse_count());
}
