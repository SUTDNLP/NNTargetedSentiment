project(MinervaSegs)

include_directories(
  basic
  /opt/mshadow/
  ../LibN3L/
)

add_definitions(-DUSE_CUDA=0)

IF(CMAKE_BUILD_TYPE MATCHES Debug)
    SET( CMAKE_CXX_FLAGS  "-w -msse3 -funroll-loops -O0" )
ELSE()
    SET( CMAKE_CXX_FLAGS  "-w -msse3 -funroll-loops -O3" )
ENDIF()
####for   openblas
add_definitions(-DMSHADOW_USE_CUDA=0)
add_definitions(-DMSHADOW_USE_CBLAS=1)
add_definitions(-DMSHADOW_USE_MKL=0)

SET( CMAKE_SHARED_LINKER_FLAGS  "-lm -lopenblas")
####endfor openblas

####for   cuda
#add_definitions(-DMSHADOW_USE_CUDA=1)
#add_definitions(-DMSHADOW_USE_CBLAS=1)
#add_definitions(-DMSHADOW_USE_MKL=0)

#SET( CMAKE_SHARED_LINKER_FLAGS  "-lm -lcudart -lcublas -lcurand" )
#include_directories(
#  $(USE_CUDA_PATH)/include
#)
#LINK_DIRECTORIES($(USE_CUDA_PATH)/lib64)
####endfor cuda

#add_subdirectory(basic)

#aux_source_directory(. DIR_SRCS)

add_executable(PoolLabeler PoolLabeler.cpp)
add_executable(PoolRNNLabeler PoolRNNLabeler.cpp)
add_executable(PoolGRNNLabeler PoolGRNNLabeler.cpp)
add_executable(PoolExLabeler PoolExLabeler.cpp)
add_executable(PoolExRNNLabeler PoolExRNNLabeler.cpp)
add_executable(PoolExGRNNLabeler PoolExGRNNLabeler.cpp)

target_link_libraries(PoolLabeler openblas)
target_link_libraries(PoolRNNLabeler openblas)
target_link_libraries(PoolGRNNLabeler openblas)
target_link_libraries(PoolExLabeler openblas)
target_link_libraries(PoolExRNNLabeler openblas)
target_link_libraries(PoolExGRNNLabeler openblas)


