cmake_minimum_required(VERSION 3.14)
project(genie-app)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Create genie-app.exe
add_executable(genie-app ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
target_include_directories(genie-app PRIVATE
     ${CMAKE_CURRENT_SOURCE_DIR}/../../..//include/Genie
     ${CMAKE_CURRENT_SOURCE_DIR}/include
     ${CMAKE_CURRENT_SOURCE_DIR}/../Genie/src/qualla/include
)

target_link_libraries(genie-app PRIVATE Genie)
