cmake_minimum_required(VERSION 3.8)
project(nonlocal_seg)
set(SRC_LIST main.cc functions/functions.cc)
add_definitions(-std=c++11)
add_definitions(-fopenmp)
add_definitions(-I/home/andres/mlpack/build/deps/ensmallen-2.10.4/include)
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})
add_subdirectory(/home/andres/dlib/ dlib_build)
find_package(Boost REQUIRED COMPONENTS system filesystem)
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})
find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

find_library(MLPACK_LIBRARY
	NAMES mlpack
	PATHS "/usr/local/lib"
)
find_path(MLPACK_INCLUDE_DIR
	NAMES mlpack/core.hpp mlpack/prereqs.hpp
	PATHS "/usr/local/include/mlpack"
)

if(MLPACK_FOUND)
	set(MLPACK_INCLUDE_DIRS ${MLPACK_INCLUDE_DIR})
	set(MLPACK_LIBRARIES ${MLPACK_LIBRARY})
endif()
include_directories(${MLPACK_INCLUDE_DIR})

add_executable(${PROJECT_NAME} ${SRC_LIST})
target_link_libraries(${PROJECT_NAME} ${ITK_LIBRARIES} ${Boost_LIBRARIES} ${ARMADILLO_LIBRARIES} ${MLPACK_LIBRARY} dlib::dlib)
