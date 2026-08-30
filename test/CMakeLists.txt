find_package(GTest REQUIRED)
find_package(glog REQUIRED)

set(link_ext_lib glog::glog GTest::gtest)
file(GLOB_RECURSE DIR_TESTS CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/test_tensor/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_op/*.cpp"
)
add_executable(test_unit
    test_main.cpp
    ${DIR_TESTS}
    ${PROJECT_SOURCE_DIR}/test/utils.cu
)

target_link_libraries(test_unit ${link_ext_lib} ${CUDART_LIBRARY})
target_include_directories(test_unit PUBLIC ${glog_INCLUDE_DIR})
target_include_directories(test_unit PUBLIC ${GTest_INCLUDE_DIR})

target_include_directories(test_unit PUBLIC ../infer/include)
target_include_directories(test_unit PRIVATE
    ${PROJECT_SOURCE_DIR}/test
    ${CUDA_INCLUDE_DIR}
)
if(TARGET qwen)
    target_link_directories(test_unit PUBLIC ${PROJECT_SOURCE_DIR}/lib)
    target_link_libraries(test_unit qwen)
endif()

set_target_properties(test_unit PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
