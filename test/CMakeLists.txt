find_package(GTest REQUIRED)

if (CUDAToolkit_FOUND)
  add_executable(lcnn_step_cuda_test lcnn_step_cuda_test.cpp)
  target_link_libraries(lcnn_step_cuda_test lcnn_step_cuda GTest::gtest gtest_main)
  gtest_discover_tests(lcnn_step_cuda_test WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
endif()