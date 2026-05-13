cmake_minimum_required(VERSION 3.20)

set(_runintgen_compiler_prefix "")
if(DEFINED Python_EXECUTABLE)
  get_filename_component(_runintgen_python_bin "${Python_EXECUTABLE}" DIRECTORY)
  get_filename_component(_runintgen_compiler_prefix
    "${_runintgen_python_bin}" DIRECTORY)
elseif(DEFINED ENV{CONDA_PREFIX})
  set(_runintgen_compiler_prefix "$ENV{CONDA_PREFIX}")
elseif(DEFINED ENV{VIRTUAL_ENV})
  set(_runintgen_compiler_prefix "$ENV{VIRTUAL_ENV}")
endif()

if(_runintgen_compiler_prefix)
  if(EXISTS "${_runintgen_compiler_prefix}/bin/clang++")
    if(NOT DEFINED CMAKE_CXX_COMPILER
        OR CMAKE_CXX_COMPILER MATCHES "^/usr/bin/clang\\+\\+$")
      set(CMAKE_CXX_COMPILER "${_runintgen_compiler_prefix}/bin/clang++"
        CACHE FILEPATH "C++ compiler" FORCE)
    endif()
  endif()
endif()
unset(_runintgen_compiler_prefix)

project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

if(NOT SKBUILD)
  message(WARNING
    "This CMake project is intended to be built through scikit-build-core. "
    "Use `pip install .` or `pip install -e .`.")
endif()

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import sys; print(sys.prefix)"
  OUTPUT_VARIABLE RUNINTGEN_PYTHON_PREFIX
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${RUNINTGEN_PYTHON_PREFIX}")

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
