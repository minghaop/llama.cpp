cmake_minimum_required(VERSION 3.31.6)

project("ai-chat" VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED true)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)

set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}"   CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" CACHE STRING "" FORCE)

# --------------------------------------------------------------------------
# AI Chat library
# --------------------------------------------------------------------------

if(DEFINED ANDROID_ABI)
    message(STATUS "Detected Android ABI: ${ANDROID_ABI}")
    if(ANDROID_ABI STREQUAL "arm64-v8a")
        set(GGML_SYSTEM_ARCH "ARM")
        set(GGML_CPU_KLEIDIAI OFF)
        set(GGML_OPENMP ON)
    elseif(ANDROID_ABI STREQUAL "x86_64")
        set(GGML_SYSTEM_ARCH "x86")
        set(GGML_CPU_KLEIDIAI OFF)
        set(GGML_OPENMP OFF)
    else()
        message(FATAL_ERROR "Unsupported ABI: ${ANDROID_ABI}")
    endif()
endif()

set(LLAMA_SRC ${CMAKE_CURRENT_LIST_DIR}/../../../../../../)
add_subdirectory(${LLAMA_SRC} build-llama)

set(TOKENIZERS_CPP_SRC ${CMAKE_CURRENT_LIST_DIR}/../../../../tokenizers-cpp)
add_subdirectory(${TOKENIZERS_CPP_SRC} build-tokenizers-cpp)

add_library(${CMAKE_PROJECT_NAME} SHARED
        ai_chat.cpp
        mnn_runtime_jni.cpp
        qwen_tokenizer_jni.cpp
        litert_native_flow_jni.cpp)

target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE
        GGML_SYSTEM_ARCH=${GGML_SYSTEM_ARCH}
        GGML_CPU_KLEIDIAI=$<BOOL:${GGML_CPU_KLEIDIAI}>
        GGML_OPENMP=$<BOOL:${GGML_OPENMP}>
)

target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}
        ${CMAKE_CURRENT_LIST_DIR}/third_party_mnn/include
        ${CMAKE_CURRENT_LIST_DIR}/third_party_mnn
        ${LLAMA_SRC}
        ${LLAMA_SRC}/common
        ${LLAMA_SRC}/include
        ${LLAMA_SRC}/ggml/include
        ${LLAMA_SRC}/ggml/src
        ${CMAKE_CURRENT_LIST_DIR}/third_party_litert_c)

set(MNN_JNILIB_DIR ${CMAKE_CURRENT_LIST_DIR}/../jniLibs/${ANDROID_ABI})

add_library(MNN SHARED IMPORTED)
set_target_properties(MNN PROPERTIES IMPORTED_LOCATION ${MNN_JNILIB_DIR}/libMNN.so)

add_library(MNN_Express SHARED IMPORTED)
set_target_properties(MNN_Express PROPERTIES IMPORTED_LOCATION ${MNN_JNILIB_DIR}/libMNN_Express.so)

add_library(llm SHARED IMPORTED)
set_target_properties(llm PROPERTIES IMPORTED_LOCATION ${MNN_JNILIB_DIR}/libllm.so)

target_link_libraries(${CMAKE_PROJECT_NAME}
        llama
        common
        tokenizers_cpp
        MNN
        MNN_Express
        llm
        android
        log
        EGL
        GLESv3
        dl)
