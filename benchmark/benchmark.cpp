#include <fstream>

#include <benchmark/benchmark.h>
#include <mkl.h>
#include <omp.h>

#include "matmul.h"
#include "utils.h"

static constexpr auto shapes_csv = "data/shapes.csv";

static void create_flop_counter(benchmark::State &state, int m, int n, int k)
{
	state.counters["FLOPS"] = benchmark::Counter(
		2.0 * m * n * k,
		benchmark::Counter::kIsIterationInvariantRate,
		benchmark::Counter::OneK::kIs1000
	);
}

static void BM_matmul(benchmark::State &state)
{
	int m = static_cast<int>(state.range(0));
	int n = static_cast<int>(state.range(1));
	int k = static_cast<int>(state.range(2));

	auto [A, B] = generate_data(m, n, k);
	auto C      = aligned_ptr(new (static_cast<std::align_val_t>(64)) float[m * n]);

	for (auto _ : state)
		matmul(A.get(), B.get(), C.get(), m, k, n);

	create_flop_counter(state, m, n, k);
}

static void BM_mkl(benchmark::State &state)
{
	int m = static_cast<int>(state.range(0));
	int n = static_cast<int>(state.range(1));
	int k = static_cast<int>(state.range(2));

	omp_set_num_threads(1);
	auto [A, B] = generate_data(m, n, k);
	auto C      = aligned_ptr(new (static_cast<std::align_val_t>(64)) float[m * n]);

	for (auto _ : state) {
		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		            m, n, k, 1.0f, A.get(), k, B.get(), n, 0.0f, C.get(), n);
	}

	create_flop_counter(state, m, n, k);
}

std::vector<int64_t> get_shape(std::string_view str)
{
	int64_t     m, n, k;
	const char *start = str.data();
	const char *end   = start + str.size();

	if (std::from_chars(start, end, m).ptr == end)
		throw std::invalid_argument("Invalid format");
	start = std::find(start, end, ',') + 1;
	if (start >= end || std::from_chars(start, end, n).ptr == end)
		throw std::invalid_argument("Invalid format");
	start = std::find(start, end, ',') + 1;
	if (start >= end || std::from_chars(start, end, k).ptr != end)
		throw std::invalid_argument("Invalid format");

	return {m, n, k};
}

int main(int argc, char **argv)
{
	benchmark::Initialize(&argc, argv);
	if (benchmark::ReportUnrecognizedArguments(argc, argv))
		return 1;

	auto b_mkl = benchmark::RegisterBenchmark(
		"Intel MKL", [](benchmark::State &state) { BM_mkl(state); });
	auto b_matmul = benchmark::RegisterBenchmark(
		"Matmul", [](benchmark::State &state) { BM_matmul(state); });

	std::ifstream file(shapes_csv);
	std::string line;
	while (std::getline(file, line)) {
		auto shape = get_shape(line);
		b_mkl->Args(shape);
		b_matmul->Args(shape);
	}

	benchmark::RunSpecifiedBenchmarks();
	benchmark::Shutdown();
}
