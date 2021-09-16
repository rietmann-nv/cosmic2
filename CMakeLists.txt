cmake_minimum_required(VERSION 3.18.0)

project(udpframereader LANGUAGES C CXX CUDA)

find_package(Python3 REQUIRED)

find_package(PythonExtensions REQUIRED)

find_package(CUDAToolkit REQUIRED)

add_library(udpframereader MODULE
  cosmicp/udpframereader.c
  cosmicp/descramble.cu
  )

set_property(TARGET udpframereader PROPERTY CUDA_ARCHITECTURES 70)

add_executable(test_cuda_descrambler
  test/test_descrambler.cpp
  cosmicp/descramble.cu
  )
set_property(TARGET test_cuda_descrambler PROPERTY CUDA_ARCHITECTURES 70)
target_link_libraries(test_cuda_descrambler
  CUDA::cudart
  )

python_extension_module(udpframereader)
install(TARGETS udpframereader LIBRARY DESTINATION cosmicp)
