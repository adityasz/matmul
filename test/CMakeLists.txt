find_package(MKL CONFIG REQUIRED PATHS $ENV{MKLROOT})
find_package(TBB REQUIRED)
find_package(GTest REQUIRED)

add_executable(matmul_test test.cpp)
target_include_directories(matmul_test
    PRIVATE ${CMAKE_SOURCE_DIR}/src
    PRIVATE ${CMAKE_SOURCE_DIR}/include
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
