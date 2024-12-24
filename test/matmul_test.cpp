#include <gtest/gtest.h>
#include <mkl.h>

#include "matmul.h"
#include "utils.h"

void compare_results(const float *actual, const float *expected, int size)
{
	for (int i = 0; i < size; ++i) {
		float epsilon = 1e-3f;
		EXPECT_NEAR(expected[i], actual[i], epsilon) << std::format(
		    "Mismatch at index {}: {:.3f} != {:.3f}.", i, actual[i], expected[i]);
	}
}

void compare(int m, int k, int n)
{
	auto [A, B]   = generate_data(m, k, n);
	auto C_mkl    = aligned_ptr(new (static_cast<std::align_val_t>(64)) float[m * n]);
	auto C_matmul = aligned_ptr(new (static_cast<std::align_val_t>(64)) float[m * n]);

	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
	            m, n, k, 1.0f, A.get(), k, B.get(), n, 0.0f, C_mkl.get(), n);

	matmul(A.get(), B.get(), C_matmul.get(), m, k, n);
	compare_results(C_matmul.get(), C_mkl.get(), m * n);
}

TEST(MatmulTest, LargeDimensions)
{
	int panel_m = 6 * 1024;
	int panel_n = 16 * 32;
	int panel_k = 1024;

	compare(2 * panel_m, 2 * panel_n, 2 * panel_k);
}

TEST(MatmulTest, EdgeCases)
{
	int k = 8;
	for (int m = 1; m <= 13; ++m) {
		for (int n = 1; n <= 33; ++n)
			compare(m, k, n);
	}
}

int main(int argc, char **argv)
{
	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
