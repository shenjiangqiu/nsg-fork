set(CMAKE_CXX_STANDARD 11)

file(GLOB_RECURSE CPP_SOURCES *.cpp)

add_library(${PROJECT_NAME} ${CPP_SOURCES})
add_library(${PROJECT_NAME}_s STATIC ${CPP_SOURCES})
target_link_libraries(${PROJECT_NAME} rust-lib-wrapper)
target_link_libraries(${PROJECT_NAME}_s rust-lib-wrapper)

#install()