cmake_minimum_required(VERSION 3.11)

project(efanna2e)
include_directories(${PROJECT_SOURCE_DIR}/include)
#OpenMP



option(ENABLE_LTO "Enable cross language linking time optimization" ON)
if(ENABLE_LTO)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT supported OUTPUT error)
  if(supported)
      message(STATUS "IPO / LTO enabled")
      set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
      add_link_options(-fuse-ld=lld)
  else()
      message(STATUS "IPO / LTO not supported: <${error}>")
  endif()
endif()


include_directories(${CMAKE_BINARY_DIR}/rust_lib)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free")
add_definitions (-std=c++17 -O3 -lboost -march=native -Wall -DINFO)
add_subdirectory(rust_lib)
add_subdirectory(src)
add_subdirectory(tests)

