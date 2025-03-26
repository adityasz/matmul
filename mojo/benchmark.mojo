import random
import time

from memory import UnsafePointer

from matmul import matmul


fn main() raises:
    var m = 6144 * 2
    var n = 512 * 2
    var k = 1024 * 2
    var a = UnsafePointer[Float32, alignment=64].alloc(m * k)
    var b = UnsafePointer[Float32, alignment=64].alloc(k * n)
    var c = UnsafePointer[Float32, alignment=64].alloc(m * n)
    
    random.randn(a, m * k, mean=0, standard_deviation=1)
    random.randn(b, k * n, mean=0, standard_deviation=1)

    matmul(a, b, c, m, k, n) # warmup
    var start: UInt = time.perf_counter_ns()
    matmul(a, b, c, m, k, n)
    var end: UInt = time.perf_counter_ns()
    print(2.0 * m * n * k / (end - start), "GFLOPS")
