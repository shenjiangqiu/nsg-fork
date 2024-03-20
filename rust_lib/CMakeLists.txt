if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CARGO_CMD cargo build)
    set(TARGET_DIR "debug")
else ()
    set(CARGO_CMD cargo build --release)
    set(TARGET_DIR "release")
endif ()

if(ENABLE_LTO)
    set(RUST_FLAGS "-Clinker-plugin-lto" "-Clinker=clang" "-Clink-arg=-fuse-ld=lld")
endif()

set(RUST_PART_LIB "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_DIR}/librust_lib.a")

set(RUST_LIB_CXX "${CMAKE_CURRENT_BINARY_DIR}/rust_lib.cpp")
add_library(rust_lib_wrapper STATIC ${RUST_LIB_CXX})
add_custom_command(
    OUTPUT ${RUST_LIB_CXX}
    COMMAND CARGO_TARGET_DIR=${CMAKE_CURRENT_BINARY_DIR} RUSTFLAGS="${RUST_FLAGS}" ${CARGO_CMD}
    COMMAND cp ${CMAKE_CURRENT_BINARY_DIR}/cxxbridge/rust_lib/src/lib.rs.cc ${RUST_LIB_CXX}
    COMMAND cp ${CMAKE_CURRENT_BINARY_DIR}/cxxbridge/rust_lib/src/lib.rs.h ${CMAKE_CURRENT_BINARY_DIR}/rust_lib.h
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(rust_lib_wrapper pthread dl ${RUST_PART_LIB})

