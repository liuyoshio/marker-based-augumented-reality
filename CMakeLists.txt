cmake_minimum_required(VERSION 3.10)
project(project)

# Set the CMake prefix path to include your custom installation path
list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew/opt")

# Find the required packages
find_package(glfw3 REQUIRED)
find_package(GLEW REQUIRED)
find_package(glm REQUIRED)
find_package(OpenCV REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})
# Add the project root directory to the include paths
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/include)

# Define your executable
add_executable(main 
    src/main.cpp

    include/global.hpp
    include/OpenGLUtils.hpp

    src/OpenGLUtils.cpp

    common/shader.cpp
    common/shader.hpp
)
# Link libraries
target_link_libraries(main ${OpenCV_LIBS} glfw GLEW::GLEW)


