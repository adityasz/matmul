cmake_minimum_required(VERSION 3.30)
project(gemm)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# TODO: Does (compiler) prefetching help?
set(CMAKE_CXX_FLAGS "-Wall -Wpedantic -Wextra -march=native -fprefetch-loop-arrays")
set(CMAKE_CXX_FLAGS_DEBUG "-g3")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ffast-math -flto")

find_package(MKL CONFIG REQUIRED PATHS $ENV{MKLROOT})
find_package(fmt CONFIG REQUIRED)
find_package(TBB REQUIRED)
find_package(benchmark REQUIRED)
find_package(GTest)
find_package(OpenMP REQUIRED)

add_library(matmul STATIC src/matmul.cpp)
target_include_directories(matmul PUBLIC include PRIVATE src)
target_link_libraries(matmul PUBLIC OpenMP::OpenMP_CXX)

add_executable(benchmark benchmark/matmul_benchmark.cpp)
target_include_directories(benchmark
	PRIVATE src
	PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
target_compile_options(benchmark
	PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
target_link_libraries(benchmark
	PRIVATE matmul
	PRIVATE benchmark::benchmark
	PRIVATE pthread
	PRIVATE TBB::tbb
	PRIVATE $<LINK_ONLY:MKL::MKL>
	PRIVATE omp
)

add_executable(matmul_test test/matmul_test.cpp)
target_include_directories(matmul_test
	PRIVATE src
	PRIVATE $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
target_compile_options(matmul_test
	PRIVATE $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
target_link_libraries(matmul_test
	PRIVATE matmul
	PRIVATE GTest::gtest
	PRIVATE GTest::gtest_main
	PRIVATE TBB::tbb
	PRIVATE $<LINK_ONLY:MKL::MKL>
	PRIVATE omp
)

enable_testing()
add_test(NAME MatmulTest COMMAND matmul_test)
