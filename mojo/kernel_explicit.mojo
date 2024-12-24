from collections import InlineArray
from memory import UnsafePointer
from sys.intrinsics import strided_load, strided_store, likely


fn kernelgen[Rows: Int, UseMask: Bool](
    sliver_a: UnsafePointer[Float32, mut=False],
    sliver_b: UnsafePointer[Float32, mut=False],
    c: UnsafePointer[Float32],
    k: Int,
    n: Int,
    mask0: SIMD[DType.bool, 8] = 0,
    mask1: SIMD[DType.bool, 8] = 0
):
    var c00: SIMD[DType.float32, 8] = 0 # Redundant initialization.
    var c01: SIMD[DType.float32, 8] = 0 # The mojo compiler complains if I
    var c10: SIMD[DType.float32, 8] = 0 # don't initialize them even though
    var c11: SIMD[DType.float32, 8] = 0 # though they are used only after
    var c20: SIMD[DType.float32, 8] = 0 # they are filled with data below
    var c21: SIMD[DType.float32, 8] = 0
    var c30: SIMD[DType.float32, 8] = 0
    var c31: SIMD[DType.float32, 8] = 0
    var c40: SIMD[DType.float32, 8] = 0
    var c41: SIMD[DType.float32, 8] = 0
    var c50: SIMD[DType.float32, 8] = 0
    var c51: SIMD[DType.float32, 8] = 0

    @parameter
    if Rows > 0:
        @parameter
        if UseMask:
            c00 = strided_load[8](c + 0 * n + 0, 1, mask0)
            c01 = strided_load[8](c + 0 * n + 8, 1, mask1)
        else:
            c00 = strided_load[8](c + 0 * n + 0, 1)
            c01 = strided_load[8](c + 0 * n + 8, 1)
    @parameter
    if Rows > 1:
        @parameter
        if UseMask:
            c10 = strided_load[8](c + 1 * n + 0, 1, mask0)
            c11 = strided_load[8](c + 1 * n + 8, 1, mask1)
        else:
            c10 = strided_load[8](c + 1 * n + 0, 1)
            c11 = strided_load[8](c + 1 * n + 8, 1)
    @parameter
    if Rows > 2:
        @parameter
        if UseMask:
            c20 = strided_load[8](c + 2 * n + 0, 1, mask0)
            c21 = strided_load[8](c + 2 * n + 8, 1, mask1)
        else:
            c20 = strided_load[8](c + 2 * n + 0, 1)
            c21 = strided_load[8](c + 2 * n + 8, 1)
    @parameter
    if Rows > 3:
        @parameter
        if UseMask:
            c30 = strided_load[8](c + 3 * n + 0, 1, mask0)
            c31 = strided_load[8](c + 3 * n + 8, 1, mask1)
        else:
            c30 = strided_load[8](c + 3 * n + 0, 1)
            c31 = strided_load[8](c + 3 * n + 8, 1)
    @parameter
    if Rows > 4:
        @parameter
        if UseMask:
            c40 = strided_load[8](c + 4 * n + 0, 1, mask0)
            c41 = strided_load[8](c + 4 * n + 8, 1, mask1)
        else:
            c40 = strided_load[8](c + 4 * n + 0, 1)
            c41 = strided_load[8](c + 4 * n + 8, 1)
    @parameter
    if Rows > 5:
        @parameter
        if UseMask:
            c50 = strided_load[8](c + 5 * n + 0, 1, mask0)
            c51 = strided_load[8](c + 5 * n + 8, 1, mask1)
        else:
            c50 = strided_load[8](c + 5 * n + 0, 1)
            c51 = strided_load[8](c + 5 * n + 8, 1)

    var a = sliver_a
    var b = sliver_b
    for _ in range(k):
        var b_vec0 = strided_load[8](b + 0, 1)
        var b_vec1 = strided_load[8](b + 8, 1)

        @parameter
        if Rows > 0:
            var a_vec = SIMD[DType.float32, 8](a[0])
            c00 = a_vec.fma(b_vec0, c00)
            c01 = a_vec.fma(b_vec1, c01)
        @parameter
        if Rows > 1:
            var a_vec = SIMD[DType.float32, 8](a[1])
            c10 = a_vec.fma(b_vec0, c10)
            c11 = a_vec.fma(b_vec1, c11)
        @parameter
        if Rows > 2:
            var a_vec = SIMD[DType.float32, 8](a[2])
            c20 = a_vec.fma(b_vec0, c20)
            c21 = a_vec.fma(b_vec1, c21)
        @parameter
        if Rows > 3:
            var a_vec = SIMD[DType.float32, 8](a[3])
            c30 = a_vec.fma(b_vec0, c30)
            c31 = a_vec.fma(b_vec1, c31)
        @parameter
        if Rows > 4:
            var a_vec = SIMD[DType.float32, 8](a[4])
            c40 = a_vec.fma(b_vec0, c40)
            c41 = a_vec.fma(b_vec1, c41)
        @parameter
        if Rows > 5:
            var a_vec = SIMD[DType.float32, 8](a[5])
            c50 = a_vec.fma(b_vec0, c50)
            c51 = a_vec.fma(b_vec1, c51)

        a += 6
        b += 16

    @parameter
    if Rows > 0:
        @parameter
        if UseMask:
            strided_store[8](c00, c + 0 * n + 0, 1, mask0)
            strided_store[8](c01, c + 0 * n + 8, 1, mask1)
        else:
            strided_store[8](c00, c + 0 * n + 0, 1)
            strided_store[8](c01, c + 0 * n + 8, 1)
    @parameter
    if Rows > 1:
        @parameter
        if UseMask:
            strided_store[8](c10, c + 1 * n + 0, 1, mask0)
            strided_store[8](c11, c + 1 * n + 8, 1, mask1)
        else:
            strided_store[8](c10, c + 1 * n + 0, 1)
            strided_store[8](c11, c + 1 * n + 8, 1)
    @parameter
    if Rows > 2:
        @parameter
        if UseMask:
            strided_store[8](c20, c + 2 * n + 0, 1, mask0)
            strided_store[8](c21, c + 2 * n + 8, 1, mask1)
        else:
            strided_store[8](c20, c + 2 * n + 0, 1)
            strided_store[8](c21, c + 2 * n + 8, 1)
    @parameter
    if Rows > 3:
        @parameter
        if UseMask:
            strided_store[8](c30, c + 3 * n + 0, 1, mask0)
            strided_store[8](c31, c + 3 * n + 8, 1, mask1)
        else:
            strided_store[8](c30, c + 3 * n + 0, 1)
            strided_store[8](c31, c + 3 * n + 8, 1)
    @parameter
    if Rows > 4:
        @parameter
        if UseMask:
            strided_store[8](c40, c + 4 * n + 0, 1, mask0)
            strided_store[8](c41, c + 4 * n + 8, 1, mask1)
        else:
            strided_store[8](c40, c + 4 * n + 0, 1)
            strided_store[8](c41, c + 4 * n + 8, 1)
    @parameter
    if Rows > 5:
        @parameter
        if UseMask:
            strided_store[8](c50, c + 5 * n + 0, 1, mask0)
            strided_store[8](c51, c + 5 * n + 8, 1, mask1)
        else:
            strided_store[8](c50, c + 5 * n + 0, 1)
            strided_store[8](c51, c + 5 * n + 8, 1)


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
        if rows == 1: kernelgen[1, False](a, b, c, k, n)
        elif rows == 2: kernelgen[2, False](a, b, c, k, n)
        elif rows == 3: kernelgen[3, False](a, b, c, k, n)
        elif rows == 4: kernelgen[4, False](a, b, c, k, n)
        elif rows == 5: kernelgen[5, False](a, b, c, k, n)
        elif likely(rows == 6): kernelgen[6, False](a, b, c, k, n)
    else:
        alias mask = InlineArray[Int8, 32](-1, -1, -1, -1, -1, -1, -1, -1,
                                           -1, -1, -1, -1, -1, -1, -1, -1,
                                            0,  0,  0,  0,  0,  0,  0,  0, 
                                            0,  0,  0,  0,  0,  0,  0,  0)
        var mask0: SIMD[DType.bool, 8] = 0 # TODO
        var mask1: SIMD[DType.bool, 8] = 0 # TODO
        if rows == 1: kernelgen[1, True](a, b, c, k, n, mask0, mask1)
        elif rows == 2: kernelgen[2, True](a, b, c, k, n, mask0, mask1)
        elif rows == 3: kernelgen[3, True](a, b, c, k, n, mask0, mask1)
        elif rows == 4: kernelgen[4, True](a, b, c, k, n, mask0, mask1)
        elif rows == 5: kernelgen[5, True](a, b, c, k, n, mask0, mask1)
        elif likely(rows == 6): kernelgen[6, True](a, b, c, k, n, mask0, mask1)
