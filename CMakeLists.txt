cmake_minimum_required(VERSION 3.8.0)
project(RBE595_Final_Project)

# Compile in Release by default
if(CMAKE_BUILD_TYPE STREQUAL "")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel ..." FORCE)
endif(CMAKE_BUILD_TYPE STREQUAL "")

# Use C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find the ARGoS package, make sure to save the ARGoS prefix
find_package(ARGoS REQUIRED)
include_directories(${CMAKE_SOURCE_DIR} ${ARGOS_INCLUDE_DIRS})
link_directories(${ARGOS_LIBRARY_DIR})
link_libraries(${ARGOS_LDFLAGS})

# Check for Buzz
find_package(Buzz REQUIRED)
include_directories(${BUZZ_C_INCLUDE_DIR})
include(${BUZZ_CMAKE_USEBUZZ}/UseBuzz.cmake)

# Compile code
add_library(MCCA SHARED MCCA.h MCCA.cpp)
target_link_libraries(MCCA
  ${BUZZ_LIBRARY}
  argos3core_simulator
  argos3plugin_simulator_dynamics2d
  argos3plugin_simulator_entities
  argos3plugin_simulator_kheperaiv
  argos3plugin_simulator_genericrobot
  argos3plugin_simulator_media
  argos3plugin_simulator_qtopengl
  argos3plugin_simulator_buzz)

# Compile Buzz files
buzz_make(MCCA.bzz)
