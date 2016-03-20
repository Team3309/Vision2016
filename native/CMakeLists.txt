cmake_minimum_required(VERSION 2.8)
project(Vision2016Native)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

file(GLOB vision_SRC
    "*.h"
    "*.cpp"
)

find_package(OpenCV REQUIRED)
add_executable(vision ${vision_SRC})
target_link_libraries(vision ${OpenCV_LIBS})