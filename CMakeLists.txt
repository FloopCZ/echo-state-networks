cmake_minimum_required(VERSION 3.3 FATAL_ERROR)

# Use clang++ by default because with g++, we could accidentaly
# fall into https://github.com/arrayfire/arrayfire/issues/2978.
set(CMAKE_CXX_COMPILER "clang++")

# Export compile commants to help VS Code.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(
  "lcnn"
  VERSION 0.2
)

# Register git hooks directory.
execute_process(
  COMMAND "git" "config" "core.hooksPath" "${CMAKE_CURRENT_SOURCE_DIR}/git-hooks"
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

# --------------
# Find Libraries
# --------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(ArrayFire REQUIRED)
find_package(Boost COMPONENTS program_options log log_setup REQUIRED)
find_package(CUDA)
find_package(Eigen3 REQUIRED)
find_package(Forge REQUIRED)
cmake_policy(SET CMP0072 NEW)
find_package(OpenGL REQUIRED)
find_package(OpenCL)

# --------------
# Compiler Flags
# --------------

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-Wall -pedantic ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
# Link all boost libraries dynamically.
add_definitions("-DBOOST_ALL_DYN_LINK")

# ----------------------
# Shorter Error Messages
# ----------------------

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set(CMAKE_CXX_FLAGS "-Wfatal-errors ${CMAKE_CXX_FLAGS}")
elseif (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  # GCC truncates multiline errors with -Wfatal-errors
  # using -fmax-erorrs instead
  set(CMAKE_CXX_FLAGS "-fmax-errors=2 ${CMAKE_CXX_FLAGS}")
endif()

# ---------------------------------------
# Utility functions for multiple backends
# ---------------------------------------

# This function adds only object file for the given executable.
# Useful for building the same executable with many different linkage options.
function(add_object EXECUTABLE_NAME)
  add_library(
    "${EXECUTABLE_NAME}" OBJECT "${EXECUTABLE_NAME}.cpp"
  )
  target_include_directories(
    "${EXECUTABLE_NAME}"
    PUBLIC ${ArrayFire_INCLUDE_DIRS}
    PUBLIC ${Boost_INCLUDE_DIRS}
    PUBLIC ${EIGEN3_INCLUDE_DIRS}
    PUBLIC ${FORGE_INCLUDE_DIRS}
  )
endfunction(add_object)

# This function links everything except for the ArrayFire backnd
# to the given object. To be able to build the final executable,
# a backend needs to be added.
file(GLOB_RECURSE all_headers "*.hpp")
function(link_without_backend EXECUTABLE_NAME EXECUTABLE_SUFFIX)
  add_executable(
    "${EXECUTABLE_NAME}${EXECUTABLE_SUFFIX}"
    "$<TARGET_OBJECTS:${EXECUTABLE_NAME}>"
    ${all_headers}
  )
  target_link_libraries(
    "${EXECUTABLE_NAME}${EXECUTABLE_SUFFIX}"
    PUBLIC cmaes
    PUBLIC "${Boost_LIBRARIES}"
    PUBLIC "${FORGE_LIBRARIES}"
    PUBLIC OpenGL::GLX
    PUBLIC tbb
  )
endfunction(link_without_backend)

# Link CPU backend to the given object. Object target has to exist.
function(link_with_cpu EXECUTABLE_NAME)
  link_without_backend("${EXECUTABLE_NAME}" "_cpu")
  target_link_libraries(
    "${EXECUTABLE_NAME}_cpu"
    PUBLIC ${ArrayFire_CPU_LIBRARIES}
  )
endfunction(link_with_cpu)

# Link CUDA backend to the given object. Object target has to exist.
function(link_with_cuda EXECUTABLE_NAME)
  link_without_backend("${EXECUTABLE_NAME}" "_cuda")
  target_link_libraries(
    "${EXECUTABLE_NAME}_cuda"
    PUBLIC ${ArrayFire_CUDA_LIBRARIES}
  )
endfunction(link_with_cuda)

# Link OpenCL backend to the given object. Object target has to exist.
function(link_with_cl EXECUTABLE_NAME)
  link_without_backend("${EXECUTABLE_NAME}" "_cl")
  target_link_libraries(
    "${EXECUTABLE_NAME}_cl"
    PUBLIC ${ArrayFire_OpenCL_LIBRARIES}
  )
endfunction(link_with_cl)

# Build the given executable with all the available backends.
function(add_with_all EXECUTABLE_NAME)
  add_object("${EXECUTABLE_NAME}")
  link_with_cpu("${EXECUTABLE_NAME}")
  if (CUDA_FOUND)
    link_with_cuda("${EXECUTABLE_NAME}")
  endif()
  if (OpenCL_FOUND)
    link_with_cl("${EXECUTABLE_NAME}")
  endif()
endfunction(add_with_all)

# ---------------------
# Build the executables
# ---------------------

add_with_all("evaluate")
add_with_all("optimize")
add_with_all("visual")
add_with_all("param_sensitivity_int")
add_with_all("param_sensitivity_double")
