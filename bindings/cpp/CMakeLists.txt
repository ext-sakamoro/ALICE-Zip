# ALICE-Zip C++ Bindings
# CMake build configuration for standalone C++ projects

cmake_minimum_required(VERSION 3.16)
project(alice_cpp VERSION 2.2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(ALICE_BUILD_EXAMPLES "Build example programs" ON)
option(ALICE_BUILD_TESTS "Build tests" ON)

# Find the native library
set(ALICE_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../libalice/target/release" CACHE PATH "Path to libalice library")

# Platform-specific library name
if(WIN32)
    set(ALICE_LIB_NAME "alice_core.dll")
    set(ALICE_IMPORT_LIB "alice_core.dll.lib")
elseif(APPLE)
    set(ALICE_LIB_NAME "libalice_core.dylib")
elseif(UNIX)
    set(ALICE_LIB_NAME "libalice_core.so")
endif()

# Create interface library for the C++ wrapper
add_library(alice_cpp INTERFACE)
target_include_directories(alice_cpp INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../libalice/include
)

# Link to native library if found
if(EXISTS "${ALICE_LIB_DIR}/${ALICE_LIB_NAME}")
    if(WIN32)
        target_link_libraries(alice_cpp INTERFACE "${ALICE_LIB_DIR}/${ALICE_IMPORT_LIB}")
    else()
        target_link_libraries(alice_cpp INTERFACE "${ALICE_LIB_DIR}/${ALICE_LIB_NAME}")
    endif()
    message(STATUS "Found libalice at: ${ALICE_LIB_DIR}/${ALICE_LIB_NAME}")
else()
    message(WARNING "libalice not found at ${ALICE_LIB_DIR}. Build libalice first with 'cargo build --release'")
endif()

# Examples
if(ALICE_BUILD_EXAMPLES)
    add_executable(perlin_example examples/perlin_example.cpp)
    target_link_libraries(perlin_example PRIVATE alice_cpp)

    add_executable(compression_example examples/compression_example.cpp)
    target_link_libraries(compression_example PRIVATE alice_cpp)
endif()

# Tests
if(ALICE_BUILD_TESTS)
    enable_testing()
    add_executable(test_alice tests/test_alice.cpp)
    target_link_libraries(test_alice PRIVATE alice_cpp)
    add_test(NAME alice_tests COMMAND test_alice)
endif()

# Installation
install(FILES alice.hpp DESTINATION include/alice)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../../libalice/include/alice.h DESTINATION include/alice)
