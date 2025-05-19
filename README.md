# Fast FP32 matmul in C++

A row-major implementation of [sgemm.c](https://salykova.github.io/matmul-cpu),
utilizing compile time metaprogramming for a 60% reduction in lines of code!

The [kernel](https://github.com/adityasz/matmul/blob/master/src/kernel.h) is
templated, and >=g++-14 does not spill registers when using
[C-style arrays](https://github.com/adityasz/matmul/blob/master/src/kernel.h#L16).

Gets ~135 GFLOPS on one 12700H core vs MKL's ~138 GFLOPS
(naive benchmark, see TODO).


## Benchmarks

Before running the benchmarks, follow
[this](https://llvm.org/docs/Benchmarking.html) to reduce system noise.

```console
$ cmake -B build/release -S . -GNinja -DCMAKE_BUILD_TYPE=Release
$ cmake --build build/release -j
$ build/release/benchmark
```


## TODO

- [ ] Run benchmarks ([properly](https://llvm.org/docs/Benchmarking.html)) and
      fine-tune hyperparameters for Alder Lake
- [ ] Parallelize (depends on benchmarking results)
- [ ] Figure out what's keeping the (near-identical) [mojo](https://github.com/modularml/mojo)
      implementation from being fast
