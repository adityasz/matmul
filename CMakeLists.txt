cmake_minimum_required(VERSION 3.30)
project(picocc)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS "-Wall -Wpedantic -Wextra -march=native")
set(CMAKE_CXX_FLAGS_DEBUG "-g3")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ffast-math -flto")

add_subdirectory(src)
add_subdirectory(benchmark)
add_subdirectory(test)
