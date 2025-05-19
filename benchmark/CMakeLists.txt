find_package(MKL CONFIG REQUIRED PATHS $ENV{MKLROOT})
find_package(TBB REQUIRED)
find_package(benchmark REQUIRED)

add_executable(benchmark benchmark.cpp)
target_include_directories(benchmark
    PRIVATE ${CMAKE_SOURCE_DIR}/src
    PRIVATE ${CMAKE_SOURCE_DIR}/include
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
