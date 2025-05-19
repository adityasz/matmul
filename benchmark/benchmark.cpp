#include <filesystem>
#include <fstream>
#include <print>

#include <benchmark/benchmark.h>
#include <mkl.h>
#include <omp.h>

#include "matmul.h"
#include "utils.h"

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
	int64_t m = 0;
	int64_t n = 0;
	int64_t k = 0;
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
	int benchmark_argc = argc;
	std::vector<char *> benchmark_argv;
	benchmark_argv.reserve(argc);
	benchmark_argv.push_back(argv[0]);

	auto shapes_path = "data/shapes.csv";
	bool found_path = false;
	for (int i = 1; i < argc; i++) {
		if (argv[i][0] != '-') {
			if (found_path) {
				std::println(stderr,
				             "Expected only one positional argument. See {} --help.",
				             argv[0]);
				exit(1);
			}
			shapes_path = argv[i];
			benchmark_argc -= 1;
			found_path = true;
		} else {
			benchmark_argv.push_back(argv[i]);
		}
	}
	benchmark::Initialize(&benchmark_argc, benchmark_argv.data(), []() {
		benchmark::PrintDefaultHelp();
		std::println(
		    "          [PATH]{: ^8}path to a CSV file containing all the shapes (m, n, k) to benchmark on\n"
		    "                {: ^8}(default: data/shapes.csv)", "", ""
		);
	});
	if (benchmark::ReportUnrecognizedArguments(benchmark_argc, benchmark_argv.data()))
		return 1;

	auto b_mkl = benchmark::RegisterBenchmark(
		"Intel MKL", [](benchmark::State &state) { BM_mkl(state); });
	auto b_matmul = benchmark::RegisterBenchmark(
		"Matmul", [](benchmark::State &state) { BM_matmul(state); });

	if (!std::filesystem::exists(shapes_path)) {
		std::println(stderr, "error: '{}' does not exist", shapes_path);
		exit(1);
	}
	std::ifstream file(shapes_path);
	std::string line;
	while (std::getline(file, line)) {
		auto shape = get_shape(line);
		b_mkl->Args(shape);
		b_matmul->Args(shape);
	}

	for (auto &b : {b_mkl, b_matmul}) {
		b->ComputeStatistics("max", [](const std::vector<double> &v) -> double {
			return *(std::ranges::max_element(v));
		});
	}

	benchmark::RunSpecifiedBenchmarks();
	benchmark::Shutdown();
}
