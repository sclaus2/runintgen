cmake_minimum_required(VERSION 3.20)

project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

if(NOT SKBUILD)
  message(WARNING
    "This CMake project is intended to be built through scikit-build-core. "
    "Use `pip install .` or `pip install -e .`.")
endif()

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import nanobind; print(nanobind.cmake_dir())"
  OUTPUT_VARIABLE RUNINTGEN_NANOBIND_CMAKE_DIR
  RESULT_VARIABLE RUNINTGEN_NANOBIND_RESULT
  ERROR_VARIABLE RUNINTGEN_NANOBIND_ERROR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT RUNINTGEN_NANOBIND_RESULT EQUAL 0)
  message(FATAL_ERROR
    "Could not query nanobind CMake directory with ${Python_EXECUTABLE}.\n"
    "${RUNINTGEN_NANOBIND_ERROR}")
endif()
list(APPEND CMAKE_PREFIX_PATH "${RUNINTGEN_NANOBIND_CMAKE_DIR}")

find_package(nanobind CONFIG REQUIRED)
find_package(Basix CONFIG REQUIRED)

nanobind_add_module(_basix_runtime
  runintgen/cpp/basix_runtime_module.cpp
)

target_compile_features(_basix_runtime PRIVATE cxx_std_20)
target_link_libraries(_basix_runtime PRIVATE Basix::basix)
target_include_directories(_basix_runtime PRIVATE runintgen/cpp)

install(TARGETS _basix_runtime LIBRARY DESTINATION runintgen)

install(
  FILES
    runintgen/cpp/runintgen_runtime_abi.h
    runintgen/cpp/runtime_basix_wrapper.h
    runintgen/cpp/runtime_basix_wrapper.cpp
    runintgen/cpp/runtime_tabulation_helpers.h
    runintgen/cpp/basix_runtime_module.cpp
  DESTINATION runintgen/cpp
)
