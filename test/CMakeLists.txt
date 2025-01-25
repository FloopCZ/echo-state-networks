find_package(GTest REQUIRED)

add_library(common_test common_test.cpp)

add_executable(lcnn_step_cpu_test lcnn_step_cpu_test.cpp)
target_link_libraries(lcnn_step_cpu_test common_test lcnn_step_cpu GTest::gtest gtest_main)
gtest_discover_tests(lcnn_step_cpu_test WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

if (CUDAToolkit_FOUND)
  add_executable(lcnn_step_cuda_test lcnn_step_cuda_test.cpp)
  target_link_libraries(lcnn_step_cuda_test common_test lcnn_step_cuda GTest::gtest gtest_main)
  gtest_discover_tests(lcnn_step_cuda_test WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
endif()

if (OpenCL_FOUND)
  add_executable(lcnn_step_cl_test lcnn_step_cl_test.cpp)
  target_link_libraries(lcnn_step_cl_test common_test lcnn_step_cl GTest::gtest gtest_main)
  gtest_discover_tests(lcnn_step_cl_test WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
endif()