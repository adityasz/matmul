# Fast FP32 matmul in C++

A row-major implementation of [matmul.c](https://salykova.github.io/matmul),
utilizing compile time metaprogramming for a 50% reduction in lines of code!

The [kernel](https://github.com/adityasz/matmul/blob/master/src/kernel.h) is
templated, and g++14 doesn't suffer from register spilling with
[C-style arrays](https://github.com/adityasz/matmul/blob/master/src/kernel.h#L16).

## TODO

- [ ] Run benchmarks ([properly](https://llvm.org/docs/Benchmark.html)) and
      fine-tune hyperparameters for Alder Lake
- [ ] Parallelize (depends on benchmarking results)
- [ ] Figure out what's keeping the (near-identical) [mojo](https://github.com/modularml/mojo)
      implementation from being fast
