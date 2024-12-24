#ifndef MATMUL_H
#define MATMUL_H


/**
 * @brief Compute the matrix-matrix product C := AB.
 *
 * The matrices A and B are assumed to be stored in row-major order and
 * the result is also stored in row-major order in `c`.
 *
 * @param a The pointer to A.
 * @param b The pointer to B.
 * @param c The pointer to C.
 * @param m The number of rows of A.
 * @param k The number of columns (rows) of A (B).
 * @param n The number of columns of B.
 */
void matmul(const float *a, const float *b, float *c, int m, int k, int n);

#endif // MATMUL_H
