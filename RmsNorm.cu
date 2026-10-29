#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

constexpr int N = 16384;
constexpr int THREADS = 128;
constexpr int WARPS = THREADS / 32;

constexpr unsigned FULL_MASK = 0xffffffffu;
constexpr float EPSILON = 1.0e-5f;

#define CUDA_CHECK(call)                                    \
   do                                                       \
   {                                                        \
      cudaError_t err = (call);                             \
      if (err != cudaSuccess)                               \
      {                                                     \
         std::cerr << "CUDA Error: "                        \
                   << cudaGetErrorString(err)               \
                   << " at " << __FILE__ << ":" << __LINE__ \
                   << std::endl;                            \
         std::exit(EXIT_FAILURE);                           \
      }                                                     \
   } while (0)

__device__ __forceinline__ float warp_reduce_sum(float value)
{
#pragma unroll
   for (int offset = 16; offset > 0; offset >>= 1)
   {
      value += __shfl_down_sync(FULL_MASK, value, offset);
   }

   return value;
}

template <int BLOCK_THREADS>
__device__ __forceinline__ float block_reduce_sum(
    float value,
    float *scratch)
{
   constexpr int NUM_WARPS = BLOCK_THREADS / 32;

   const int lane = threadIdx.x & 31;
   const int warp = threadIdx.x >> 5;

   // Reduce within each warp.
   value = warp_reduce_sum(value);

   // Each warp leader stores its partial sum.
   if (lane == 0)
   {
      scratch[warp] = value;
   }

   __syncthreads();

   // First warp reduces the warp partial sums.
   if (warp == 0)
   {
      value = (lane < NUM_WARPS) ? scratch[lane] : 0.0f;
      value = warp_reduce_sum(value);

      if (lane == 0)
      {
         scratch[0] = value;
      }
   }

   __syncthreads();

   // All threads receive the complete block sum.
   return scratch[0];
}

template <int BLOCK_THREADS>
__global__ void rmsnorm_kernel(
    const float *__restrict__ input,
    const float *__restrict__ weight,
    float *__restrict__ output,
    int rows,
    float eps)
{
   static_assert(BLOCK_THREADS % 32 == 0,
                 "BLOCK_THREADS must be a multiple of 32");

   static_assert(N % BLOCK_THREADS == 0,
                 "N must be divisible by BLOCK_THREADS");

   constexpr int ITEMS = N / BLOCK_THREADS;

   const int row = blockIdx.x;
   const int tid = threadIdx.x;

   if (row >= rows)
   {
      return;
   }

   const size_t base = static_cast<size_t>(row) * N;

   const float *x = input + base;
   float *y = output + base;

   __shared__ float scratch[BLOCK_THREADS / 32];

   // Load elements and accumulate their squares.
   float values[ITEMS];
   float local_sum_sq = 0.0f;

#pragma unroll
   for (int i = 0; i < ITEMS; ++i)
   {
      const int idx = tid + i * BLOCK_THREADS;

      const float v = x[idx];

      values[i] = v;
      local_sum_sq = fmaf(v, v, local_sum_sq);
   }

   // Reduce the sum of squares across the whole block.
   const float sum_sq =
       block_reduce_sum<BLOCK_THREADS>(local_sum_sq, scratch);

   // Calculate inverse RMS.
   const float mean_sq = sum_sq / static_cast<float>(N);
   const float inv_rms = rsqrtf(mean_sq + eps);

   // Normalize, apply learned scale, and store.
#pragma unroll
   for (int i = 0; i < ITEMS; ++i)
   {
      const int idx = tid + i * BLOCK_THREADS;

      y[idx] = (values[i] * inv_rms) * weight[idx];
   }
}

void launch_rmsnorm(
    const float *input,
    const float *weight,
    float *output,
    int rows,
    float eps = EPSILON,
    cudaStream_t stream = nullptr)
{
   rmsnorm_kernel<THREADS>
       <<<rows, THREADS, 0, stream>>>(
           input, weight, output, rows, eps);

   CUDA_CHECK(cudaGetLastError());
}

void cpu_rmsnorm(
    const std::vector<float> &input,
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
         const double v = input[base + j];
         sum_sq += v * v;
      }

      const double mean_sq = sum_sq / N;
      const double inv_rms = 1.0 / std::sqrt(mean_sq + eps);

      for (int j = 0; j < N; ++j)
      {
         output[base + j] = static_cast<float>(
             static_cast<double>(input[base + j]) * inv_rms * weight[j]);
      }
   }
}

void generate_random_data(
    std::vector<float> &input,
    std::vector<float> &weight,
    unsigned int seed)
{
   std::mt19937 rng(seed);

   std::normal_distribution<float> input_dist(0.0f, 1.0f);
   std::normal_distribution<float> weight_dist(0.0f, 1.0f);

   for (float &v : input)
   {
      v = input_dist(rng);
   }

   for (float &v : weight)
   {
      v = weight_dist(rng);
   }
}

struct ErrorStats
{
   float max_abs_error = 0.0f;
   float max_rel_error = 0.0f;
   bool finite = true;
};

ErrorStats validate_output(
    const std::vector<float> &actual,
    const std::vector<float> &reference)
{
   ErrorStats stats;

   for (size_t i = 0; i < actual.size(); ++i)
   {
      const float a = actual[i];
      const float b = reference[i];

      if (!std::isfinite(a))
      {
         stats.finite = false;
         continue;
      }

      const float abs_error = std::fabs(a - b);
      const float rel_error =
          abs_error / std::max(std::fabs(b), 1.0e-6f);

      stats.max_abs_error =
          std::max(stats.max_abs_error, abs_error);

      stats.max_rel_error =
          std::max(stats.max_rel_error, rel_error);
   }

   return stats;
}

float benchmark_gpu_ms(
    const float *d_input,
    const float *d_weight,
    float *d_output,
    int rows,
    float eps,
    int warmup = 20,
    int iterations = 100)
{
   for (int i = 0; i < warmup; ++i)
   {
      launch_rmsnorm(
          d_input, d_weight, d_output, rows, eps);
   }

   CUDA_CHECK(cudaDeviceSynchronize());

   cudaEvent_t start, stop;

   CUDA_CHECK(cudaEventCreate(&start));
   CUDA_CHECK(cudaEventCreate(&stop));

   CUDA_CHECK(cudaEventRecord(start));

   for (int i = 0; i < iterations; ++i)
   {
      launch_rmsnorm(
          d_input, d_weight, d_output, rows, eps);
   }

   CUDA_CHECK(cudaGetLastError());
   CUDA_CHECK(cudaEventRecord(stop));
   CUDA_CHECK(cudaEventSynchronize(stop));

   float elapsed_ms = 0.0f;

   CUDA_CHECK(cudaEventElapsedTime(
       &elapsed_ms, start, stop));

   CUDA_CHECK(cudaEventDestroy(start));
   CUDA_CHECK(cudaEventDestroy(stop));

   return elapsed_ms / iterations;
}

int main(int argc, char **argv)
{
   float *d_input = nullptr;
   float *d_weight = nullptr;
   float *d_output = nullptr;

   try
   {
      int rows = 128;
      unsigned int seed = 12345;
      float eps = EPSILON;

      if (argc >= 2)
      {
         rows = std::stoi(argv[1]);
      }

      if (argc >= 3)
      {
         seed = static_cast<unsigned int>(
             std::stoul(argv[2]));
      }

      if (argc >= 4)
      {
         eps = std::stof(argv[3]);
      }

      if (rows <= 0)
      {
         throw std::invalid_argument(
             "rows must be positive");
      }

      if (!(eps > 0.0f) || !std::isfinite(eps))
      {
         throw std::invalid_argument(
             "epsilon must be finite and positive");
      }

      CUDA_CHECK(cudaSetDevice(0));

      cudaDeviceProp prop{};

      CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

      if (THREADS > prop.maxThreadsPerBlock)
      {
         throw std::runtime_error(
             "THREADS exceed device block limit");
      }

      const size_t count =
          static_cast<size_t>(rows) * N;

      const size_t input_bytes =
          count * sizeof(float);

      const size_t weight_bytes =
          static_cast<size_t>(N) * sizeof(float);

      std::vector<float> h_input(count);
      std::vector<float> h_weight(N);
      std::vector<float> h_output(count);
      std::vector<float> h_reference(count);

      generate_random_data(
          h_input, h_weight, seed);

      // Allocate device memory.
      CUDA_CHECK(cudaMalloc(
          reinterpret_cast<void **>(&d_input),
          input_bytes));

      CUDA_CHECK(cudaMalloc(
          reinterpret_cast<void **>(&d_weight),
          weight_bytes));

      CUDA_CHECK(cudaMalloc(
          reinterpret_cast<void **>(&d_output),
          input_bytes));

      // Copy inputs to the GPU.
      CUDA_CHECK(cudaMemcpy(
          d_input,
          h_input.data(),
          input_bytes,
          cudaMemcpyHostToDevice));

      CUDA_CHECK(cudaMemcpy(
          d_weight,
          h_weight.data(),
          weight_bytes,
          cudaMemcpyHostToDevice));

      // Run kernel once for correctness testing.
      launch_rmsnorm(
          d_input, d_weight, d_output, rows, eps);

      CUDA_CHECK(cudaDeviceSynchronize());

      CUDA_CHECK(cudaMemcpy(
          h_output.data(),
          d_output,
          input_bytes,
          cudaMemcpyDeviceToHost));

      // Calculate CPU reference.
      cpu_rmsnorm(
          h_input,
          h_weight,
          h_reference,
          rows,
          eps);

      // Validate results.
      const ErrorStats stats =
          validate_output(h_output, h_reference);

      constexpr float ATOL = 1.0e-5f;
      constexpr float RTOL = 1.0e-4f;

      bool close = true;

      for (size_t i = 0; i < count; ++i)
      {
         const float a = h_output[i];
         const float b = h_reference[i];

         if (!std::isfinite(a) ||
             std::fabs(a - b) >
                 ATOL + RTOL * std::fabs(b))
         {
            close = false;
            break;
         }
      }

      const bool correct = stats.finite && close;

      std::cout << "------------------------\n";
      std::cout << "CUDA RMSNorm\n";
      std::cout << "------------------------\n";

      std::cout << "GPU: " << prop.name << '\n';
      std::cout << "Rows: " << rows << '\n';
      std::cout << "Row Length: " << N << '\n';
      std::cout << "Data Type: FP32\n";
      std::cout << "Threads / Block: "
                << THREADS << '\n';
      std::cout << "Epsilon: " << eps << '\n';
      std::cout << "Seed: " << seed << '\n';

      std::cout << std::scientific
                << std::setprecision(6);

      std::cout << "\nCorrectness\n";
      std::cout << "All finite: "
                << (stats.finite ? "YES" : "NO")
                << '\n';

      std::cout << "Max abs error: "
                << stats.max_abs_error << '\n';

      std::cout << "Max rel error: "
                << stats.max_rel_error << '\n';

      std::cout << "Result: "
                << (correct ? "PASS" : "FAIL")
                << "\n\n";

      if (correct)
      {
         const float latency_ms =
             benchmark_gpu_ms(
                 d_input,
                 d_weight,
                 d_output,
                 rows,
                 eps);

         if (!(latency_ms > 0.0f) ||
             !std::isfinite(latency_ms))
         {
            throw std::runtime_error(
                "Invalid GPU benchmark latency");
         }

         const double logical_bytes =
             static_cast<double>(input_bytes) * 2.0 + static_cast<double>(weight_bytes);

         const double bandwidth_gbs =
             logical_bytes /
             (static_cast<double>(latency_ms) * 1.0e-3) / 1.0e9;

         const double elements_per_second =
             static_cast<double>(count) /
             (static_cast<double>(latency_ms) * 1.0e-3);

         std::cout << "Performance\n";
         std::cout << "Kernel Latency: "
                   << latency_ms << " ms\n";

         std::cout << "Logical BW: "
                   << bandwidth_gbs << " GB/s\n";

         std::cout << "Elements/sec: "
                   << elements_per_second << '\n';
      }
      else
      {
         std::cout
             << "Benchmark skipped because validation failed.\n";
      }

      CUDA_CHECK(cudaFree(d_input));
      d_input = nullptr;

      CUDA_CHECK(cudaFree(d_weight));
      d_weight = nullptr;

      CUDA_CHECK(cudaFree(d_output));
      d_output = nullptr;

      return correct ? EXIT_SUCCESS : EXIT_FAILURE;
   }
   catch (const std::exception &e)
   {
      std::cerr << "Error: " << e.what() << '\n';

      if (d_input != nullptr)
         cudaFree(d_input);

      if (d_weight != nullptr)
         cudaFree(d_weight);

      if (d_output != nullptr)
         cudaFree(d_output);

      return EXIT_FAILURE;
   }
}
