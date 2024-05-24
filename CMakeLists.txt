cmake_minimum_required(VERSION 3.11)

project(efanna2e)
include_directories(${PROJECT_SOURCE_DIR}/include)
#OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(FATAL_ERROR "no OpenMP supprot")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free")
add_definitions (-std=c++14 -lboost -march=native -Wall)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# add_subdirectory(rust-crates/rust_lib)
add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(unit_tests)
