#include <cuda_runtime.h>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <algorithm>
#include <random>
#include <stdexcept>
#include <vector>

constexpr int N = 16383;
constexpr int THREADS = 512;
constexpr int WARPS = THREADS / 32;
constexpr float EPSILON = 1.0e-5f;

#define CUDA_CHECK(call)                                                                                          \
   do                                                                                                             \
   {                                                                                                              \
      cudaError_t err = (call);                                                                                   \
      if (err != cudaSuccess)                                                                                     \
      {                                                                                                           \
         std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
         std::exit(EXIT_FAILURE);                                                                                 \
      }                                                                                                           \
   } while (0)

__device__ __forceinline__ float warp_reduce_sum(float value)
{
   for (int offset = 16; offset > 0; offset >>= 1)
   {
      value += __shfl_down_sync(0xffffffffu, value, offset);
   }
   return value;
}

__device__ __forceinline__ float block_reduce_sum(float value, float *scratch)
{
   const int lane = threadIdx.x & 31;
   const int warp = threadIdx.x >> 5;

   value = warp_reduce_sum(value);

   if (lane == 0)
   {
      scratch[warp] = value;
   }

   __syncthreads();

   if (warp == 0)
   {
      value = (lane < WARPS) ? scratch[lane] : 0.0f;
      value = warp_reduce_sum(value);

      if (lane == 0)
      {
         scratch[0] = value;
      }
   }
   __syncthreads();
   return scratch[0];
}

template <int BLOCK_THREADS>
__global__ void residual_rmsnorm_kernel(
    const float *__restrict__ input,
    const float *__restrict__ residual,
    const float *__restrict__ weight,
    float *__restrict__ output,
    int rows,
    float eps)
{
   constexpr int ITEMS = (N + BLOCK_THREADS - 1) / BLOCK_THREADS;

   const int row = blockIdx.x;
   const int tid = threadIdx.x;

   if (row >= rows)
      return;

   const size_t base = static_cast<size_t>(row) * N;

   const float *x = input + base;
   const float *r = residual + base;
   float *y = output + base;

   __shared__ float scratch[BLOCK_THREADS / 32];

   float values[ITEMS];
   float local_sum_sq = 0.0f;

#pragma unroll
   for (int i = 0; i < ITEMS; ++i)
   {
      const int idx = tid + i * BLOCK_THREADS;
      if (idx < N)
      {
         const float u = x[idx] + r[idx];
         values[i] = u;
         local_sum_sq = fmaf(u, u, local_sum_sq);
      }
   }

   const float sum_sq = block_reduce_sum(local_sum_sq, scratch);
   const float mean_sq = sum_sq / static_cast<float>(N);
   const float inv_rms = rsqrtf(mean_sq + eps);

#pragma unroll
   for (int i = 0; i < ITEMS; ++i)
   {
      const int idx = tid + i * BLOCK_THREADS;
      if (idx < N)
      {
         y[idx] = (values[i] * inv_rms) * weight[idx];
      }
   }
}

__global__ void residual_add_kernel(
    const float *input,
    const float *residual,
    float *intermediate,
    size_t count)
{
   const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
   if (idx < count)
   {
      intermediate[idx] = input[idx] + residual[idx];
   }
}

template <int BLOCK_THREADS>
__global__ void rmsnorm_kernel(
    const float *__restrict__ input,
    const float *__restrict__ weight,
    float *__restrict__ output,
    int rows,
    float eps)
{
   constexpr int ITEMS = (N + BLOCK_THREADS - 1) / BLOCK_THREADS;

   const int row = blockIdx.x;
   const int tid = threadIdx.x;
   if (row >= rows)
      return;

   const size_t base = static_cast<size_t>(row) * N;

   const float *x = input + base;
   float *y = output + base;

   __shared__ float scratch[BLOCK_THREADS / 32];

   float values[ITEMS];
   float local_sum_sq = 0.0f;

#pragma unroll
   for (int i = 0; i < ITEMS; ++i)
   {
      const int idx = tid + i * BLOCK_THREADS;
      if (idx < N)
      {
         const float v = x[idx];
         values[i] = v;
         local_sum_sq = fmaf(v, v, local_sum_sq);
      }
   }

   const float sum_sq = block_reduce_sum(local_sum_sq, scratch);
   const float inv_rms = rsqrtf(sum_sq / static_cast<float>(N) + eps);

#pragma unroll
   for (int i = 0; i < ITEMS; ++i)
   {
      const int idx = tid + i * BLOCK_THREADS;
      if (idx < N)
      {
         y[idx] = (values[i] * inv_rms) * weight[idx];
      }
   }
}

void launch_fused(
    const float *input,
    const float *residual,
    const float *weight,
    float *output,
    int rows,
    float eps)
{
   residual_rmsnorm_kernel<THREADS><<<rows, THREADS>>>(input, residual, weight, output, rows, eps);

   CUDA_CHECK(cudaGetLastError());
}

void launch_unfused(
    const float *input,
    const float *residual,
    const float *weight,
    float *intermediate,
    float *output,
    int rows,
    float eps)
{
   const size_t count = static_cast<size_t>(rows) * N;
   constexpr int ADD_THREADS = 256;

   const int add_blocks = static_cast<int>((count + ADD_THREADS - 1) / ADD_THREADS);

   residual_add_kernel<<<add_blocks, ADD_THREADS>>>(input, residual, intermediate, count);

   CUDA_CHECK(cudaGetLastError());

   rmsnorm_kernel<THREADS><<<rows, THREADS>>>(intermediate, weight, output, rows, eps);

   CUDA_CHECK(cudaGetLastError());
}

void generate_random_data(
    std::vector<float> &input,
    std::vector<float> &resiudal,
    std::vector<float> &weight,
    unsigned int seed)
{
   std::mt19937 rng(seed);

   std::normal_distribution<float> input_dist(0.0f, 1.0f);
   std::normal_distribution<float> residual_dist(0.0f, 1.0f);
   std::normal_distribution<float> weight_dist(0.0f, 1.0f);

   for (float &v : input)
      v = input_dist(rng);
   for (float &v : resiudal)
      v = residual_dist(rng);
   for (float &v : weight)
      v = weight_dist(rng);
}

void cpu_refernce(
    const std::vector<float> &input,
    const std::vector<float> &residual,
    const std::vector<float> &weight,
    std::vector<float> &output,
    int rows,
    float eps)
{
   for (int row = 0; row < rows; ++row)
   {
      const size_t base = static_cast<size_t>(row) * N;
      double sum_sq = 0.0;

      for (int j = 0; j < N; ++j)
      {
         const double u =
             static_cast<double>(input[base + j]) +
             static_cast<double>(residual[base + j]);

         sum_sq += u * u;
      }
      const double inv_rms = 1.0 / std::sqrt(sum_sq / N + static_cast<double>(eps));

      for (int j = 0; j < N; ++j)
      {
         const double u =
             static_cast<double>(input[base + j]) +
             static_cast<double>(residual[base + j]);

         output[base + j] = static_cast<float>(
             u * inv_rms * static_cast<double>(weight[j]));
      }
   }
}

struct ErrorStats
{
   float max_abs = 0.0f;
   float max_rel = 0.0f;
   bool all_finite = true;
   bool passed = true;
};

ErrorStats validate(
    const std::vector<float> &actual,
    const std::vector<float> &reference)
{
   constexpr float ATOL = 1.0e-5f;
   constexpr float RTOL = 1.0e-4f;

   ErrorStats stats;

   for (size_t i = 0; i < actual.size(); ++i)
   {
      const float a = actual[i];
      const float b = reference[i];

      if (!std::isfinite(a))
      {
         stats.all_finite = false;
         stats.passed = false;
         continue;
      }

      const float abs_err = std::fabs(a - b);
      const float rel_err = abs_err / std::max(std::fabs(b), 1.0e-6f);

      stats.max_abs = std::max(stats.max_abs, abs_err);
      stats.max_rel = std::max(stats.max_rel, rel_err);

      if (abs_err > ATOL + RTOL * std::fabs(b))
      {
         stats.passed = false;
      }
   }
   return stats;
}

template <typename Fn>
float benchmark_gpu_ms(
    Fn fn,
    int warmup = 20,
    int iterations = 100)
{
   for (int i = 0; i < warmup; ++i)
   {
      fn();
   }

   CUDA_CHECK(cudaDeviceSynchronize());

   cudaEvent_t start, stop;
   CUDA_CHECK(cudaEventCreate(&start));
   CUDA_CHECK(cudaEventCreate(&stop));

   CUDA_CHECK(cudaEventRecord(start));

   for (int i = 0; i < iterations; ++i)
   {
      fn();
   }

   CUDA_CHECK(cudaEventRecord(stop));
   CUDA_CHECK(cudaEventSynchronize(stop));

   float elapsed_ms = 0.0f;
   CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

   CUDA_CHECK(cudaEventDestroy(start));
   CUDA_CHECK(cudaEventDestroy(stop));

   return elapsed_ms / iterations;
}

int main(
    int argc,
    char **argv)
{
   try
   {
      int rows = 128;
      unsigned int seed = 12345;
      float eps = EPSILON;

      if (argc >= 2)
         rows = std::stoi(argv[1]);
      if (argc >= 3)
         seed = static_cast<unsigned int>(std::stoul(argv[2]));

      if (argc >= 4)
         eps = std::stof(argv[3]);

      if (rows <= 0)
         throw std::invalid_argument("rows must be positive");

      if (!(eps > 0.0f) || !std::isfinite(eps))
         throw std::invalid_argument("epsilon must be finite and positive");

      CUDA_CHECK(cudaSetDevice(0));

      cudaDeviceProp prop{};
      CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

      if (THREADS > prop.maxThreadsPerBlock)
         throw std::runtime_error("THREADS exceeds device block limit");

      const size_t count = static_cast<size_t>(rows) * N;
      const size_t data_bytes = count * sizeof(float);
      const size_t weight_bytes = N * sizeof(float);

      std::vector<float> h_input(count);
      std::vector<float> h_residual(count);
      std::vector<float> h_weight(N);
      std::vector<float> h_fused(count);
      std::vector<float> h_unfused(count);
      std::vector<float> h_reference(count);

      generate_random_data(
          h_input, h_residual, h_weight, seed);

      float *d_input = nullptr;
      float *d_residual = nullptr;
      float *d_weight = nullptr;
      float *d_intermediate = nullptr;
      float *d_fused = nullptr;
      float *d_unfused = nullptr;

      CUDA_CHECK(cudaMalloc(&d_input, data_bytes));
      CUDA_CHECK(cudaMalloc(&d_residual, data_bytes));
      CUDA_CHECK(cudaMalloc(&d_weight, weight_bytes));
      CUDA_CHECK(cudaMalloc(&d_intermediate, data_bytes));
      CUDA_CHECK(cudaMalloc(&d_fused, data_bytes));
      CUDA_CHECK(cudaMalloc(&d_unfused, data_bytes));

      CUDA_CHECK(cudaMemcpy(
          d_input, h_input.data(), data_bytes, cudaMemcpyHostToDevice));

      CUDA_CHECK(cudaMemcpy(
          d_residual, h_residual.data(), data_bytes, cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(
          d_weight, h_weight.data(), weight_bytes, cudaMemcpyHostToDevice));

      // Run both implementations for correctness.
      launch_fused(
          d_input, d_residual, d_weight, d_fused, rows, eps);

      launch_unfused(
          d_input, d_residual, d_weight,
          d_intermediate, d_unfused, rows, eps);

      // Wait for the GPU and copy the actual results to the host.
      CUDA_CHECK(cudaDeviceSynchronize());

      CUDA_CHECK(cudaMemcpy(
          h_fused.data(), d_fused, data_bytes,
          cudaMemcpyDeviceToHost));

      CUDA_CHECK(cudaMemcpy(
          h_unfused.data(), d_unfused, data_bytes,
          cudaMemcpyDeviceToHost));

      // Validate only after copying the GPU outputs.
      cpu_refernce(
          h_input,
          h_residual,
          h_weight,
          h_reference,
          rows,
          eps);
      const ErrorStats fused_stats = validate(h_fused, h_reference);
      const ErrorStats unfused_stats = validate(h_unfused, h_reference);

      std::cout << "----------" << std::endl;
      std::cout << "CUDA Residual with RMSNORM" << std::endl;
      std::cout << "----------" << std::endl;

      std::cout << "GPU: " << prop.name << "\n";
      std::cout << "Rows: " << rows << "\n";
      std::cout << "Row length: " << N << "\n";
      std::cout << "Data Type:   FP32 \n";
      std::cout << "Threads/Block: " << THREADS << "\n";
      std::cout << "Epsilon: " << eps << "\n";
      std::cout << "Random Seed: " << seed << "\n";

      std::cout << std::scientific << std::setprecision(6);

      std::cout << "Correctness\n";
      std::cout << "Fused max absolute error: " << fused_stats.max_abs << "\n";
      std::cout << "Fused max relative error: " << fused_stats.max_rel << "\n";
      std::cout << "Fused finite outputs: " << (fused_stats.all_finite ? "PASS" : "FAIL") << "\n";
      std::cout << "Fused Results: " << (fused_stats.passed ? "PASS" : "FAIL") << "\n";

      std::cout << "Unfused max absolute error: " << unfused_stats.max_abs << "\n";
      std::cout << "Unfused result: " << (unfused_stats.passed ? "PASS" : "FAIL") << "\n";

      // Benchmark fused implementation
      const float fused_ms = benchmark_gpu_ms([&]()
                                              { launch_fused(
                                                    d_input, d_residual, d_weight, d_fused, rows, eps); });

      const float unfused_ms = benchmark_gpu_ms([&]()
                                                { launch_unfused(d_input, d_residual, d_weight, d_intermediate, d_unfused, rows, eps); });

      const double logical_bytes = 3.0 * static_cast<double>(data_bytes);

      const double bandwidth_gbs = logical_bytes / (fused_ms * 1.0e6);

      const double elements_per_second = static_cast<double>(count) / (fused_ms * 1.0e-3);

      std::cout << "Performance:\n";
      std::cout << "Fused Latency: " << fused_ms << " ms\n";
      std::cout << "Un-Fused Latency: " << unfused_ms << " ms\n";
      std::cout << "Estimated BW: " << bandwidth_gbs << " GB/s\n";
      std::cout << "Elements Per Second: " << elements_per_second << "\n";

      if (fused_ms > 0.0f)
      {
         std::cout << "Fused speedup: " << unfused_ms / fused_ms << "x\n";
      }

      CUDA_CHECK(cudaFree(d_input));
      CUDA_CHECK(cudaFree(d_residual));
      CUDA_CHECK(cudaFree(d_weight));
      CUDA_CHECK(cudaFree(d_intermediate));
      CUDA_CHECK(cudaFree(d_fused));
      CUDA_CHECK(cudaFree(d_unfused));

      return (fused_stats.passed && unfused_stats.passed) ? EXIT_SUCCESS : EXIT_FAILURE;
   }
   catch (const std::exception &e)
   {
      std::cerr << "Error: " << e.what() << "\n";
      return EXIT_FAILURE;
   }
}