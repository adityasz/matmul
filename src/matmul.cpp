/**
 * A and B are divided into blocks of size `panel_m` by `panel_k` and `panel_k`
 * by `panel_n`. Each block is further divided into slivers. Each sliver of A
 * has `reg_block_m` rows (except maybe the slivers in the last row) and each
 * sliver of B has `reg_block_n` columns (except maybe the slivers in the last
 * column).
 */

#include "matmul.h"
#include "kernel.h"

#include <algorithm>
#include <cstring>

static constexpr int reg_block_m = 6;
static constexpr int reg_block_n = 16;

// TODO: find good numbers for Alderlake
static constexpr int panel_m = 6144;
static constexpr int panel_n = 512;
static constexpr int panel_k = 1024;

[[gnu::aligned(64)]] static float block_a[panel_m * panel_k];
[[gnu::aligned(64)]] static float block_b[panel_k * panel_n];

/**
 * Packs a sliver of the matrix A into `block_a`.
 *
 * The elements of a sliver from A are read along the columns, which are read
 * from left to right.
 */
[[gnu::always_inline]] inline
void pack_sliver_a(const float *a, float *sliver_a, int rows, int cols, int k)
{
	for (int j = 0; j < cols; j++) {
		for (int i = 0; i < rows; i++)
			*sliver_a++ = a[i * k + j];
		for (int i = rows; i < reg_block_m; i++)
			*sliver_a++ = 0;
	}
}

/// Packs a block of the matrix A into `block_a`.
[[gnu::always_inline]] inline
void pack_block_a(const float *a, float *block_a, int rows, int cols, int k)
{
	for (int i = 0; i < rows; i += reg_block_m) {
		int sliver_rows = std::min(reg_block_m, rows - i);
		pack_sliver_a(&a[i * k], &block_a[i * cols], sliver_rows, cols, k);
	}
}

/**
 * Packs a sliver of the matrix B into `block_b`.
 *
 * The elements of a sliver from B are read along the rows, which are read
 * from top to bottom.
 */
[[gnu::always_inline]] inline
void pack_sliver_b(const float *b, float *sliver_b, int rows, int cols, int n)
{
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++)
			*sliver_b++ = b[i * n + j];
		for (int j = cols; j < reg_block_n; j++)
			*sliver_b++ = 0;
	}
}

/// Packs a block of the matrix B into `block_b`.
[[gnu::always_inline]] inline
void pack_block_b(const float *b, float *block_b, int rows, int cols, int n)
{
	for (int j = 0; j < cols; j += reg_block_n) {
		int sliver_cols = std::min(reg_block_n, cols - j);
		pack_sliver_b(&b[j], &block_b[j * rows], rows, sliver_cols, n);
	}
}

void matmul(const float *a, const float *b, float *c, int m, int k, int n)
{
	std::memset(c, 0, sizeof(float) * m * n);

	for (int i_p = 0; i_p < m; i_p += panel_m) {
		int m_b = std::min(panel_m, m - i_p);
		for (int l_p = 0; l_p < k; l_p += panel_k) {
			int k_b = std::min(panel_k, k - l_p);
			pack_block_a(&a[i_p * k + l_p], block_a, m_b, k_b, k);
			for (int j_p = 0; j_p < n; j_p += panel_n) {
				int n_b = std::min(panel_n, n - j_p);
				pack_block_b(&b[l_p * n + j_p], block_b, k_b, n_b, n);
				for (int i = 0; i < m_b; i += reg_block_m) {
					for (int j = 0; j < n_b; j += reg_block_n) {
						int rows = std::min(reg_block_m, m_b - i);
						int cols = std::min(reg_block_n, n_b - j);
						kernel(&block_a[i * k_b],
						       &block_b[k_b * j],
						       &c[(i_p + i) * n + (j_p + j)],
						       rows, cols, k_b, n);
					}
				}
			}
		}
	}
}
