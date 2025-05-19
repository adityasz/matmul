find_package(OpenMP REQUIRED)

add_library(matmul STATIC matmul.cpp)
target_include_directories(matmul 
    PUBLIC ${CMAKE_SOURCE_DIR}/include 
    PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(matmul PUBLIC OpenMP::OpenMP_CXX)
