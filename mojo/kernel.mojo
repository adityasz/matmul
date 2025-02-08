from collections import InlineArray, InlineList
from memory import UnsafePointer, bitcast
from sys.intrinsics import strided_load, strided_store, likely


@always_inline
fn kernel_inner[Rows: Int, UseMask: Bool](
    sliver_a: UnsafePointer[Float32, mut=False],
    sliver_b: UnsafePointer[Float32, mut=False],
    c: UnsafePointer[Float32],
    k: Int,
    n: Int,
    mask0: SIMD[DType.bool, 8] = 0,
    mask1: SIMD[DType.bool, 8] = 0
):
    var a: UnsafePointer[Float32, mut=False] = sliver_a
    var b: UnsafePointer[Float32, mut=False] = sliver_b
    var c_reg = InlineArray[SIMD[DType.float32, 8], 12](unsafe_uninitialized=True)

    @parameter
    for i in range(6):
        @parameter
        if Rows > i:
            @parameter
            if UseMask:
                c_reg[i * 2 + 0] = strided_load[8](c + i * n + 0, 1, mask0)
                c_reg[i * 2 + 1] = strided_load[8](c + i * n + 8, 1, mask1)
            else:
                c_reg[i * 2 + 0] = strided_load[8](c + i * n + 0, 1)
                c_reg[i * 2 + 1] = strided_load[8](c + i * n + 8, 1)

    for _ in range(k):
        var b_vec0 = strided_load[8](b + 0, 1)
        var b_vec1 = strided_load[8](b + 8, 1)
        @parameter
        for i in range(6):
            @parameter
            if Rows > i:
                var a_vec = SIMD[DType.float32, 8](a[i])
                c_reg[i * 2 + 0] = a_vec.fma(b_vec0, c_reg[i * 2 + 0])
                c_reg[i * 2 + 1] = a_vec.fma(b_vec1, c_reg[i * 2 + 1])
        a += 6
        b += 16

    @parameter
    for i in range(6):
        @parameter
        if Rows > i:
            @parameter
            if UseMask:
                strided_store[8](c_reg[i * 2 + 0], c + i * n + 0, 1, mask0)
                strided_store[8](c_reg[i * 2 + 1], c + i * n + 8, 1, mask1)
            else:
                strided_store[8](c_reg[i * 2 + 0], c + i * n + 0, 1)
                strided_store[8](c_reg[i * 2 + 1], c + i * n + 8, 1)


@always_inline
fn kernel(
    a: UnsafePointer[Float32, mut=False],
    b: UnsafePointer[Float32, mut=False],
    c: UnsafePointer[Float32],
    rows: Int,
    cols: Int,
    k: Int,
    n: Int
):
    if (likely(cols == 16)):
        if likely(rows == 6): kernel_inner[6, False](a, b, c, k, n)
        elif rows == 1: kernel_inner[1, False](a, b, c, k, n)
        elif rows == 2: kernel_inner[2, False](a, b, c, k, n)
        elif rows == 3: kernel_inner[3, False](a, b, c, k, n)
        elif rows == 4: kernel_inner[4, False](a, b, c, k, n)
        elif rows == 5: kernel_inner[5, False](a, b, c, k, n)
    else:
        alias mask = InlineArray[SIMD[DType.bool, 1], 32](1, 1, 1, 1, 1, 1, 1, 1,
                                                          1, 1, 1, 1, 1, 1, 1, 1,
                                                          0, 0, 0, 0, 0, 0, 0, 0, 
                                                          0, 0, 0, 0, 0, 0, 0, 0)
        var mask0 = strided_load[8](mask.unsafe_ptr() + 16 - cols + 0, 1)
        var mask1 = strided_load[8](mask.unsafe_ptr() + 16 - cols + 8, 1)
        if likely(rows == 6): kernel_inner[6, True](a, b, c, k, n, mask0, mask1)
        elif rows == 1: kernel_inner[1, True](a, b, c, k, n, mask0, mask1)
        elif rows == 2: kernel_inner[2, True](a, b, c, k, n, mask0, mask1)
        elif rows == 3: kernel_inner[3, True](a, b, c, k, n, mask0, mask1)
        elif rows == 4: kernel_inner[4, True](a, b, c, k, n, mask0, mask1)
        elif rows == 5: kernel_inner[5, True](a, b, c, k, n, mask0, mask1)
