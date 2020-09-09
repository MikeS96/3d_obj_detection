cmake_minimum_required (VERSION 3.5.1)
project (FEATURE_EXTRACTION)
find_package (PCL 1.8.1 REQUIRED)

#set(PCL_INCLUDE_DIRS /usr/local/include/pcl-1.8)

link_libraries(stdc++fs)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
add_executable(vfh src/vfh.cpp)
target_link_libraries(vfh ${PCL_LIBRARIES})

