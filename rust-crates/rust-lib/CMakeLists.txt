include(FetchContent)

FetchContent_Declare(
    Corrosion
    GIT_REPOSITORY https://github.com/corrosion-rs/corrosion.git
    GIT_TAG v0.4 # Optionally specify a commit hash, version tag or branch here
)
FetchContent_MakeAvailable(Corrosion)

# Import targets defined in a package or workspace manifest `Cargo.toml` file
corrosion_import_crate(MANIFEST_PATH ./Cargo.toml)

add_library(rust-lib-wrapper INTERFACE)
target_link_libraries(rust-lib-wrapper INTERFACE rust-lib)
target_include_directories(rust-lib-wrapper INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/)