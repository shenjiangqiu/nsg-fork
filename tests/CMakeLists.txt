set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(test_nsg_index test_nsg_index.cpp)
target_link_libraries(test_nsg_index ${PROJECT_NAME} -ltcmalloc)

add_executable(test_nsg_search test_nsg_search.cpp)
target_link_libraries(test_nsg_search ${PROJECT_NAME})

add_executable(test_nsg_optimized_search test_nsg_optimized_search.cpp)
target_link_libraries(test_nsg_optimized_search ${PROJECT_NAME} -ltcmalloc)

add_executable(sjq_test_nsg_index sjq_test_nsg_index.cpp)
target_link_libraries(sjq_test_nsg_index ${PROJECT_NAME})

add_executable(test_nn test_nn.cpp)
target_link_libraries(test_nn ${PROJECT_NAME})
add_executable(fix_nn fix_nn.cpp)
target_link_libraries(fix_nn ${PROJECT_NAME})


add_executable(test_traversal test_traversal.cpp)
target_link_libraries(test_traversal ${PROJECT_NAME})

add_executable(test1 task_demo.cpp)