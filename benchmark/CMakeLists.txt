include(ExternalProject)

if (DEFINED ENV{XDG_CACHE_HOME})
	set(DEPS_CACHE "$ENV{XDG_CACHE_HOME}/cmake_deps")
else()
	set(DEPS_CACHE "$ENV{HOME}/.cache/cmake_deps")
endif()

ExternalProject_Add(csv_parser
	GIT_REPOSITORY https://github.com/ben-strasser/fast-cpp-csv-parser
	SOURCE_DIR ${DEPS_CACHE}/github/ben-strasser/fast-cpp-csv-parser
	BINARY_DIR "" # header-only
	BUILD_COMMAND ""
	CONFIGURE_COMMAND ""
	INSTALL_COMMAND "")

find_package(MKL CONFIG REQUIRED PATHS $ENV{MKLROOT})
find_package(TBB REQUIRED)
find_package(benchmark REQUIRED)

add_executable(benchmark benchmark.cpp)
add_dependencies(benchmark csv_parser)
target_include_directories(benchmark
    PRIVATE ${CMAKE_SOURCE_DIR}/src
    PRIVATE ${CMAKE_SOURCE_DIR}/include
	PRIVATE ${DEPS_CACHE}/github/ben-strasser/fast-cpp-csv-parser
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
