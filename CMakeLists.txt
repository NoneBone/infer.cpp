cmake_minimum_required(VERSION 3.16)
project(infer_cpp LANGUAGES CXX CUDA)

# CUDA headers are required by the tensor tests (the project does not compile
# CUDA sources as part of this target yet).
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-12.8" CACHE PATH "CUDA toolkit root")
set(CUDA_INCLUDE_DIR "${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/include")
find_library(CUDART_LIBRARY cudart
    PATHS "${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/lib"
    REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
find_package(glog REQUIRED)
find_package(Armadillo REQUIRED)
find_package(absl REQUIRED)
find_package(nlohmann_json REQUIRED)
find_library(RE2_LIBRARY re2 REQUIRED)

aux_source_directory(infer/source/base/ DIR_BASE)
aux_source_directory(infer/source/model/ DIR_MODEL)
aux_source_directory(infer/source/op/ DIR_OP)
aux_source_directory(infer/source/op/kernels DIR_KERNEL)
aux_source_directory(infer/source/tensor/ DIR_TENSOR)
aux_source_directory(infer/source/sampler DIR_SAMPLE)


file(GLOB_RECURSE DIR_KERNEL
    "${PROJECT_SOURCE_DIR}/infer/source/op/kernels/*.cpp"
    "${PROJECT_SOURCE_DIR}/infer/source/op/kernels/*.cu"
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
find_package(GTest REQUIRED)
set(QWEN_SOURCES ${DIR_TENSOR} ${DIR_BASE} ${DIR_OP} ${DIR_MODEL} ${DIR_KERNEL} ${DIR_SAMPLE})
if(QWEN_SOURCES)
    add_library(qwen SHARED ${QWEN_SOURCES})
    target_link_libraries(qwen
        PRIVATE
            sentencepiece
            glog::glog
            absl::base
            absl::strings
            gtest
            gtest_main
            pthread
            ${RE2_LIBRARY}
            ${CUDART_LIBRARY}
            ${ARMADILLO_LIBRARIES}
        PUBLIC
            nlohmann_json::nlohmann_json
    )
    target_include_directories(qwen PUBLIC ${glog_INCLUDE_DIR})
    target_include_directories(qwen PUBLIC ${PROJECT_SOURCE_DIR}/infer/include)
    target_include_directories(qwen PUBLIC ${Armadillo_INCLUDE_DIR})
    target_include_directories(qwen PUBLIC ${CUDA_INCLUDE_DIR})
    # sentencepiece_processor.h provides a fallback absl::string_view unless
    # it is told to use the same installed Abseil headers as this project.
    target_compile_definitions(qwen PUBLIC _USE_TF_STRING_VIEW)
endif()


add_executable(infer_cpp test/test_main.cpp)
target_link_directories(infer_cpp PUBLIC ${PROJECT_SOURCE_DIR}/lib)
if(TARGET qwen)
    target_link_libraries(infer_cpp qwen)
endif()
add_subdirectory(test)
