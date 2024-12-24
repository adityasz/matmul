#include <benchmark/benchmark.h>
#include <mkl.h>
#include <omp.h>

#include "matmul.h"
#include "utils.h"

// 3 panels along each dimension
static constexpr int m = 6144 * 3;
static constexpr int n = 512 * 3;
static constexpr int k = 1024 * 3;

static void BM_matmul(benchmark::State &state) {
	auto [A, B] = generate_data(m, k, n);
	auto C      = aligned_ptr(new (static_cast<std::align_val_t>(64)) float[m * n]);

	for (auto _ : state)
		matmul(A.get(), B.get(), C.get(), m, k, n);

	state.counters["FLOPS"] = benchmark::Counter(
		2.0 * m * n * k,
		benchmark::Counter::kIsIterationInvariantRate,
		benchmark::Counter::OneK::kIs1000
	);
}

static void BM_mkl(benchmark::State &state) {
	omp_set_num_threads(1);
	auto [A, B] = generate_data(m, k, n);
	auto C      = aligned_ptr(new (static_cast<std::align_val_t>(64)) float[m * n]);

	for (auto _ : state) {
		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		            m, n, k, 1.0f, A.get(), k, B.get(), n, 0.0f, C.get(), n);
	}

	state.counters["FLOPS"] = benchmark::Counter(
	    2.0 * m * n * k,
	    benchmark::Counter::kIsIterationInvariantRate,
	    benchmark::Counter::OneK::kIs1000
	);
}

BENCHMARK(BM_mkl);
BENCHMARK(BM_matmul);

BENCHMARK_MAIN();
