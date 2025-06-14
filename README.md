# Fast FP32 matmul in C++

A row-major implementation of [sgemm.c](https://salykova.github.io/matmul-cpu),
utilizing compile time metaprogramming for a 60% reduction in lines of code!

The [kernel](https://github.com/adityasz/matmul/blob/master/src/kernel.h) is
templated, and >=g++-14 does not spill registers when using
[C-style arrays](https://github.com/adityasz/matmul/blob/master/src/kernel.h#L16).

Gets ~97% of the speed of Intel MKL on certain shapes (single-threaded); see TODO.

## TODO

- Tune block sizes for Alder Lake
- Parallelize with OpenMP
