import math
import random

from memory import UnsafePointer, memset_zero

from matmul import matmul


fn naive(
    a: UnsafePointer[Float32, mut=False],
    b: UnsafePointer[Float32, mut=False],
    c: UnsafePointer[Float32],
    m: Int,
    k: Int,
    n: Int
):
    memset_zero(c, m * n)
    for l in range(k):
        for i in range(m):
            for j in range(n):
                c[i * n + j] += a[i * k + l] * b[l * n + j]


fn print_matrix(a: UnsafePointer[Float32, mut=False], m: Int, n: Int):
    for i in range(m):
        for j in range(n):
            var x = round(a[i * n + j] * 100) / 100
            print(x, end='\t')
        print()

fn compare(m: Int, n: Int, k:Int):
    var a = UnsafePointer[Float32, alignment=64].alloc(m * k)
    var b = UnsafePointer[Float32, alignment=64].alloc(k * n)
    var c_matmul = UnsafePointer[Float32, alignment=64].alloc(m * n)
    var c_naive = UnsafePointer[Float32, alignment=64].alloc(m * n)

    random.seed(0)
    random.rand(a, m * k, min=-1e2, max=1e2)
    random.rand(b, k * n, min=-1e2, max=1e2)

    naive(a, b, c_naive, m, k, n)
    matmul(a, b, c_matmul, m, k, n)

    for i in range(m):
        for j in range(n):
            if not math.isclose(c_matmul[i * n + j], c_naive[i * n + j], atol=1e-3):
                print("Mismatch at (", i, ", ", j, "): ",
                      c_matmul[i * n + j], " != ", c_naive[i * n + j],
                      ". m = ", m, ", k = ", k, ", n = ", n)
                return
    
    a.free()
    b.free()
    c_matmul.free()
    c_naive.free()


fn main():
    var m = 6144 * 2
    var k = 1024 * 2
    var n = 512 * 2
    compare(m, k, n)

    for m in range(13):
        for n in range(33):
            compare(m, n, 8)
