#ifndef UTILS_H
#define UTILS_H

#include <algorithm>
#include <execution>
#include <memory>
#include <random>

struct aligned_deleter {
	void operator()(float *p) const
	{
		operator delete[](p, static_cast<std::align_val_t>(64));
	}
};

using aligned_ptr = std::unique_ptr<float[], aligned_deleter>;

inline auto generate_data(int m, int n, int k) -> std::pair<aligned_ptr, aligned_ptr>
{
	auto A = aligned_ptr(new (static_cast<std::align_val_t>(64)) float[m * k]);
	auto B = aligned_ptr(new (static_cast<std::align_val_t>(64)) float[k * n]);

	auto rand_float = []() -> float {
		thread_local std::mt19937 gen(std::random_device{}());
		std::normal_distribution  dis(0.0f, 1.0f);
		return dis(gen);
	};

	std::generate_n(std::execution::par_unseq, A.get(), m * k, rand_float);
	std::generate_n(std::execution::par_unseq, B.get(), k * n, rand_float);

	return {std::move(A), std::move(B)};
}

#endif // UTILS_H
