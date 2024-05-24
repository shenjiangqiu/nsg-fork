build_release_cmake:
    cmake -B build_release -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=On
build_release:build_release_cmake
    cmake --build build_release -- -j

build_debug_cmake:
    cmake -B build_debug -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=On
build_debug:build_debug_cmake
    cmake --build build_debug -- -j

run_release fdata knndata k c l out:build_release
    ./build_release/tests/test_nsg_index {{fdata}} {{knndata}} {{k}} {{c}} {{l}} {{out}}


test_release name="":build_release
    ./build_release/unit_tests/tests "{{name}}"