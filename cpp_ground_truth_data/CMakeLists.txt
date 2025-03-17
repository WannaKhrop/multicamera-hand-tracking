# path to cmake
set(CMAKE_TOOLCHAIN_FILE "C:/Users/khrop/Desktop/robot/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")

# Find required packages
find_package(argparse REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(Franka CONFIG REQUIRED)
find_package(franka-proxy CONFIG REQUIRED)
find_package(Threads REQUIRED)


# Add source files to this project
file(GLOB_RECURSE GESTURE_PROGRAMMING_SOURCE_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp *.hpp *.c *.h)

# Define the main executable
add_executable(gestureprogramming "main.cpp")
target_sources(gestureprogramming PRIVATE ${GESTURE_PROGRAMMING_SOURCE_FILES})

# Link required libraries to the executable
target_link_libraries(gestureprogramming
  PRIVATE 
    Eigen3::Eigen
    Franka::Franka
    Franka::libfranka-common
    franka_control
    franka_proxy_share
    franka_proxy_client
    Threads::Threads
)

# Function to organize source files in IDEs (e.g., Visual Studio)
function(assign_source_group)
    foreach(_source IN ITEMS ${ARGN})
        if (IS_ABSOLUTE "${_source}")
            file(RELATIVE_PATH _source_rel "${CMAKE_CURRENT_SOURCE_DIR}" "${_source}")
        else()
            set(_source_rel "${_source}")
        endif()
        get_filename_component(_source_path "${_source_rel}" PATH)
        string(REPLACE "/" "\\" _source_path_msvc "${_source_path}")
        source_group("${_source_path_msvc}" FILES "${_source}")
    endforeach()
endfunction(assign_source_group)

# Assign source groups for the files
assign_source_group(${GESTURE_PROGRAMMING_SOURCE_FILES})