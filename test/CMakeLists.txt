find_package(GTest REQUIRED)
find_package(glog REQUIRED)

set(link_ext_lib glog::glog GTest::gtest)
aux_source_directory(../test/test_tensor DIR_TEST_TENSOR)
add_executable(test_unit test_main.cpp ${DIR_TEST_TENSOR}) #   test_math.cpp

target_link_libraries(test_unit ${link_ext_lib})
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
