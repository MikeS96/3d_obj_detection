cmake_minimum_required (VERSION 3.5.1)
project (FEATURE_EXTRACTION)
find_package (PCL 1.8.1 REQUIRED)

link_libraries(stdc++fs)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
add_executable(pcl_features src/pcl_features.cpp)
target_link_libraries(pcl_features ${PCL_LIBRARIES})

