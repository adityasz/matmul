#ifndef KERNEL_H
#define KERNEL_H

#include <immintrin.h>

template<int Rows, bool UseMask>
inline __attribute__((always_inline))
static void kernelgen(const float *a,
                      const float *b,
                      float *c,
                      int k,
                      int n,
                      __m256i mask0 = __m256i(),
                      __m256i mask1 = __m256i())
{
	__m256 c_reg[6][2];
	for (int i = 0; i < 6; i++) {
		if (Rows > i) {
			if constexpr (UseMask) {
				c_reg[i][0] = _mm256_maskload_ps(&c[i * n + 0], mask0);
				c_reg[i][1] = _mm256_maskload_ps(&c[i * n + 8], mask1);
			} else {
				c_reg[i][0] = _mm256_loadu_ps(&c[i * n + 0]);
				c_reg[i][1] = _mm256_loadu_ps(&c[i * n + 8]);
			}
		}
	}
	for (int p = 0; p < k; p++) {
		__m256 b_vec0 = _mm256_load_ps(&b[0]);
		__m256 b_vec1 = _mm256_load_ps(&b[8]);
		for (int i = 0; i < 6; i++) {
			if (Rows > i) {
				__m256 a_vec = _mm256_broadcast_ss(&a[i]);
				c_reg[i][0] = _mm256_fmadd_ps(a_vec, b_vec0, c_reg[i][0]);
				c_reg[i][1] = _mm256_fmadd_ps(a_vec, b_vec1, c_reg[i][1]);
			}
		}
		a += 6;
		b += 16;
	}
	for (int i = 0; i < 6; i++) {
		if (Rows > i) {
			if constexpr (UseMask) {
				_mm256_maskstore_ps(&c[i * n + 0], mask0, c_reg[i][0]);
				_mm256_maskstore_ps(&c[i * n + 8], mask1, c_reg[i][1]);
			} else {
				_mm256_storeu_ps(&c[i * n + 0], c_reg[i][0]);
				_mm256_storeu_ps(&c[i * n + 8], c_reg[i][1]);
			}
		}
	}
}

/**
 * @brief The micro-kernel that computes C_ := A_ B_ + C_.
 *
 * A_ and B_ are slivers from the matrices A and B with dimensions `rows` by `k`
 * and `k` by `cols` and C_ is the micro-block with dimensions `rows` by `cols`
 * in C, where `rows` <= 6 and `cols` <= 16.
 */
inline __attribute__((always_inline))
void kernel(const float *a,
            const float *b,
            float *c,
            int rows,
            int cols,
            int k,
            int n)
{
	[[likely]]
	if (cols == 16) {
		switch (rows) {
			[[likely]]
			case 6: kernelgen<6, false>(a, b, c, k, n); break;
			case 1: kernelgen<1, false>(a, b, c, k, n); break;
			case 2: kernelgen<2, false>(a, b, c, k, n); break;
			case 3: kernelgen<3, false>(a, b, c, k, n); break;
			case 4: kernelgen<4, false>(a, b, c, k, n); break;
			case 5: kernelgen<5, false>(a, b, c, k, n); break;
		    }
	} else {
        __attribute__((aligned(64)))
        static int32_t mask[32] = {-1, -1, -1, -1, -1, -1, -1, -1,
                                   -1, -1, -1, -1, -1, -1, -1, -1,
                                    0,  0,  0,  0,  0,  0,  0,  0,
                                    0,  0,  0,  0,  0,  0,  0,  0};
		__m256i mask0 = _mm256_loadu_si256(reinterpret_cast<__m256i *>(&mask[16 - cols + 0]));
		__m256i mask1 = _mm256_loadu_si256(reinterpret_cast<__m256i *>(&mask[16 - cols + 8]));
		switch (rows) {
			[[likely]]
			case 6: kernelgen<6, true>(a, b, c, k, n, mask0, mask1); break;
			case 1: kernelgen<1, true>(a, b, c, k, n, mask0, mask1); break;
			case 2: kernelgen<2, true>(a, b, c, k, n, mask0, mask1); break;
			case 3: kernelgen<3, true>(a, b, c, k, n, mask0, mask1); break;
			case 4: kernelgen<4, true>(a, b, c, k, n, mask0, mask1); break;
			case 5: kernelgen<5, true>(a, b, c, k, n, mask0, mask1); break;
		}
	}
}

#endif // KERNEL_H
