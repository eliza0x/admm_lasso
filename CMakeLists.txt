# cmake_minimum_required(VERSION 3.15)
project(admm_lasso)

set(CMAKE_CXX_STANDARD 11)

if(NOT EIGEN_DIR)
  set(EIGEN_DIR ./include)  
  message("Eigen Path isn't set, so EIGEN_DIR is set Default_Path=./include")
  message("If you want to change the Path, enter command like \"cmake . -DEIGEN_DIR=/path/to/Eigen -DBUILD_TARGET=LOCAL -DCMAKE_BUILD_TYPE=Release \"")
else()
  message("Now Eigen Path is EIGEN_PATH=${EIGEN_DIR}")  
endif()

include_directories(${EIGEN_DIR})

add_executable(admm_lasso main.cpp)
