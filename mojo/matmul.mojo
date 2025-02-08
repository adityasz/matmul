from collections import InlineArray
from memory import UnsafePointer, memset_zero, memset
from sys.intrinsics import strided_load, strided_store
from sys.info import sizeof

from kernel import kernel


alias reg_block_m = 6
alias reg_block_n = 16
alias panel_m = reg_block_m * 1024
alias panel_n = reg_block_n * 32
alias panel_k = 1024


fn pack_sliver_a(
    a: UnsafePointer[Float32, mut=False],
    sliver_a: UnsafePointer[Float32],
    rows: Int,
    cols: Int,
    k: Int
):
    var sliver = sliver_a
    for j in range(cols):
        for i in range(rows):
            sliver[] = a[i * k + j]
            sliver += 1
        memset_zero(sliver, reg_block_m - rows)
        sliver += reg_block_m - rows


fn pack_block_a(
    a: UnsafePointer[Float32, mut=False],
    block_a: UnsafePointer[Float32],
    rows: Int,
    cols: Int,
    k: Int
):
    for i in range(0, rows, reg_block_m):
        var sliver_rows = min(reg_block_m, rows - i)
        pack_sliver_a(a + i * k, block_a + i * cols, sliver_rows, cols, k)


fn pack_sliver_b(
    b: UnsafePointer[Float32, mut=False],
    sliver_b: UnsafePointer[Float32],
    rows: Int,
    cols: Int,
    n: Int
):
    var sliver = sliver_b
    for i in range(rows):
        for j in range(cols):
            sliver[] = b[i * n + j]
            sliver += 1
        memset_zero(sliver, reg_block_n - cols)
        sliver += reg_block_n - cols


fn pack_block_b(
    b: UnsafePointer[Float32, mut=False],
    block_b: UnsafePointer[Float32],
    rows: Int,
    cols: Int,
    n: Int
):
    for j in range(0, cols, reg_block_n):
        var sliver_cols = min(reg_block_n, cols - j)
        pack_sliver_b(b + j, block_b + j * rows, rows, sliver_cols, n)


fn matmul(
    a: UnsafePointer[Float32, mut=False],
    b: UnsafePointer[Float32, mut=False],
    c: UnsafePointer[Float32],
    m: Int,
    k: Int,
    n: Int
):
    var block_a = UnsafePointer[Float32, alignment=64].alloc(panel_m * panel_k)
    var block_b = UnsafePointer[Float32, alignment=64].alloc(panel_k * panel_n)

    memset_zero(c, m * n)
    for i_p in range(0, m, panel_m):
        var m_c = min(panel_m, m - i_p)
        for l_p in range(0, k, panel_k):
            var k_c = min(panel_k, k - l_p)
            pack_block_a(a + i_p * k + l_p, block_a, m_c, k_c, k)
            for j_p in range(0, n, panel_n):
                var n_c = min(panel_n, n - j_p)
                pack_block_b(b + l_p * n + j_p, block_b, k_c, n_c, n)
                for i in range(0, m_c, reg_block_m):
                    for j in range(0, n_c, reg_block_n):
                        var rows = min(reg_block_m, m_c - i)
                        var cols = min(reg_block_n, n_c - j)
                        kernel(block_a + i * k_c,
                               block_b + k_c * j,
                               c + (i_p + i) * n + (j_p + j),
                               rows, cols, k_c, n)

    block_a.free()
    block_b.free()
