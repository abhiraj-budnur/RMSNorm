Terminal Outputs

The following results were collected on an NVIDIA GeForce MX330 using FP32 data, 128 rows, and an epsilon of 1.0e-5.

1. RMSNorm

Configuration: Row length = 16,384, threads per block = 128.

text
------------------------
CUDA RMSNorm
------------------------
GPU: NVIDIA GeForce MX330
Rows: 128
Row Length: 16384
Data Type: FP32
Threads / Block: 128
Epsilon: 1e-05
Seed: 12345

Correctness
All finite: YES
Max abs error: 1.907349e-06
Max rel error: 2.489220e-07
Result: PASS

Performance
Kernel Latency: 3.822080e-01 ms
Logical BW: 4.406698e+01 GB/s
Elements/sec: 5.486939e+09


Observations

The implementation passed the numerical validation.
Maximum absolute error was approximately 
1.91
×
10
−
6
.
Maximum relative error was approximately 
2.49
×
10
−
7
.
Average kernel latency was approximately 0.3822 ms.
Estimated logical memory bandwidth was approximately 44.07 GB/s.
Throughput was approximately 5.49 billion elements per second.
2. Fused and Unfused Residual RMSNorm

Configuration: Row length = 16,383, threads per RMSNorm block = 512.

text
----------
CUDA Residual with RMSNORM
----------
GPU: NVIDIA GeForce MX330
Rows: 128
Row length: 16383
Data Type:   FP32
Threads/Block: 512
Epsilon: 1e-05
Random Seed: 12345
Correctness
Fused max absolute error: 1.907349e-06
Fused max relative error: 3.070883e-07
Fused finite outputs: PASS
Fused Results: PASS
Unfused max absolute error: 1.907349e-06
Unfused result: PASS
Performance:
Fused Latency: 6.564249e-01 ms
Un-Fused Latency: 9.885184e-01 ms
Estimated BW: 3.833536e+01 GB/s
Elements Per Second: 3.194614e+09
Fused speedup: 1.505912e+00x


Observations

The reported fused and unfused implementations both passed validation.
Maximum absolute error for the fused implementation was approximately 
1.91
×
10
−
6
.
Maximum relative error for the fused implementation was approximately 
3.07
×
10
−
7
.
Fused latency was approximately 0.6564 ms.
Unfused latency was approximately 0.9885 ms.
The fused implementation achieved a reported speedup of approximately 1.51×.
Benchmark Results Summary

All performance measurements below are from the NVIDIA GeForce MX330.

Metric	RMSNorm	Fused Residual RMSNorm	Unfused Residual RMSNorm
Row length	16,384	16,383	16,383
Threads per block	128	512	512
Maximum absolute error	1.907349e-06	1.907349e-06	1.907349e-06
Average latency (ms)	0.382208	0.656425	0.988518
Reported bandwidth (GB/s)	44.067	38.335	Not reported
Elements/sec	5.487 billion	3.195 billion	Not reported
Correctness result	PASS	PASS	PASS

Note: The standalone RMSNorm workload has a different row length and thread-block configuration, so its performance is not a direct comparison with residual RMSNorm.

Fused vs. Unfused Performance Analysis

The fused residual RMSNorm kernel combines residual addition and normalization in a single kernel launch. The unfused implementation performs these operations using two separate kernels, with an intermediate tensor written to and read from global memory.

The measured speedup is:

Speedup
=
0.988518
0.656425
≈
1.506

Therefore, the fused implementation is approximately 1.51× faster than the unfused implementation for this measured workload.

The reduction in latency is:

Latency reduction
=
(
1
−
0.656425
0.988518
)
×
100
≈
33.6
%

This corresponds to approximately 33.6% lower execution latency for the fused implementation.

The results are consistent with the expected benefits of kernel fusion: eliminating an intermediate global-memory write and read, and reducing kernel launch overhead.

These results are specific to the tested GPU, tensor dimensions, CUDA configuration, and benchmark methodology. Additional measurements across different row counts and thread-block sizes would be needed to determine how the speedup scales.

Conclusion

These experiments demonstrate two CUDA implementations of RMSNorm-related operations:

Standalone RMSNorm: Achieved a measured latency of approximately 0.3822 ms and reported logical bandwidth of 44.07 GB/s.
Fused Residual RMSNorm: Combined residual addition and normalization into a single kernel, achieving a measured latency of approximately 0.6564 ms.
Unfused Residual RMSNorm: Used separate residual addition and RMSNorm kernels, with a measured latency of approximately 0.9885 ms.

For the tested residual RMSNorm workload, kernel fusion reduced latency by approximately 33.6%, corresponding to a speedup of 1.51×.

The experiments illustrate how CUDA warp reductions, shared memory, register reuse, and kernel fusion can be used to implement and optimize GPU tensor operations.