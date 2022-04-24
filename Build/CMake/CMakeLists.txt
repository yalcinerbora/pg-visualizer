cmake_minimum_required(VERSION 3.20)

# All CMake Related files is in the current folder
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}")

# Only Debug and Release
set(CMAKE_CONFIGURATION_TYPES Debug;Release CACHE STRING "Selected by user")
#set(CMAKE_DEFAULT_BUILD_TYPE Release CACHE STRING "Selected by user")
set(CMAKE_BUILD_TYPE Release CACHE STRING "Selected by user")

# Globals Common for All Projects
# TODO: Align CUDA Standard with CXX Standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CUDA_RUNTIME_LIBRARY Shared)

project(PGVisualizer LANGUAGES CXX)

# CMAKE Project Definitions and Source, Library Folders are separate
# Directories
set(MRAY_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../..)
set(MRAY_SOURCE_DIRECTORY ${MRAY_DIRECTORY}/Source)
set(MRAY_LIB_DIRECTORY ${MRAY_DIRECTORY}/Lib)
set(MRAY_LIB_INCLUDE_DIRECTORY ${MRAY_DIRECTORY}/Lib/Include)
set(MRAY_BIN_DIRECTORY ${MRAY_DIRECTORY}/Bin)
set(MRAY_RESOURCE_DIRECTORY ${MRAY_DIRECTORY}/Resources)
# Build is probably not used but its here for convenience
set(MRAY_BUILD_DIRECTORY ${MRAY_DIRECTORY}/Build)
# Working Dir is used for debugging (currently shaders are here so...)
# For Visual Studio Projects this is copied to WorkingDir property for executables
set(MRAY_WORKING_DIRECTORY ${MRAY_DIRECTORY}/WorkingDir)

# Determine Platform and Config
if(MSVC)
    set(MRAY_PLATFORM_NAME Win)
    set(MRAY_PREPROCESSOR_DEFS_GENERIC ${MRAY_PREPROCESSOR_DEFS_GENERIC}
        -D_UNICODE
        -DUNICODE
        -DNOMINMAX)
    # Filter ALL_BUILD and ZERO_CHECK projects
    set_property(GLOBAL PROPERTY USE_FOLDERS ON)
    set(PREDEFINED_TARGETS_FOLDER "CustomTargets")
elseif(APPLE)
    set(MRAY_PLATFORM_NAME Mac)
elseif(UNIX)
    set(MRAY_PLATFORM_NAME Linux)
    include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
else()
    message(FATAL_ERROR "Unknown platform... Terminating CMake.")
endif()

# Force Colored Output on Ninja
if ((CMAKE_GENERATOR MATCHES "Ninja") AND UNIX)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        add_compile_options(-fcolor-diagnostics)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        add_compile_options(-fdiagnostics-color=always)
    endif()
endif()

# Generic Compile Options
if(MSVC)
    # Warnings
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/W3>
                        $<$<COMPILE_LANGUAGE:CXX>:/Zi>
                        $<$<COMPILE_LANGUAGE:CXX>:/external:W0>)
    add_link_options($<$<COMPILE_LANGUAGE:CXX>:/DEBUG>)
else()
    # Clang / GCC both use same syntax
    # Warnings
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall>
                        $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
                        $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>
                        $<$<COMPILE_LANGUAGE:CXX>:-g3>)
endif()

# Platform Specific Lib Bin Folders
set(MRAY_PLATFORM_LIB_DIRECTORY ${MRAY_LIB_DIRECTORY}/${MRAY_PLATFORM_NAME})
set(MRAY_PLATFORM_BIN_DIRECTORY ${MRAY_BIN_DIRECTORY}/${MRAY_PLATFORM_NAME})
# Platform & Configurations Related Lib Bin folders
set(MRAY_CONFIG_LIB_DIRECTORY ${MRAY_PLATFORM_LIB_DIRECTORY}/$<CONFIG>)
set(MRAY_CONFIG_BIN_DIRECTORY ${MRAY_PLATFORM_BIN_DIRECTORY}/$<CONFIG>)
# Set cmake vars for output
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${MRAY_CONFIG_BIN_DIRECTORY})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${MRAY_CONFIG_BIN_DIRECTORY})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${MRAY_CONFIG_BIN_DIRECTORY})
# Meta library directory
# All precompiled libraries are resides here
link_directories(${MRAY_CONFIG_LIB_DIRECTORY}
                 ${MRAY_PLATFORM_LIB_DIRECTORY})

# Generic Preprocessor Definitions
set(MRAY_PREPROCESSOR_DEFS_GENERIC
    ${MRAY_PREPROCESSOR_DEFS_GENERIC}
    $<$<CONFIG:Debug>:METU_DEBUG>
    $<$<CONFIG:Release>:NDEBUG>)

set(MRAY_PREPROCESSOR_DEFS_CUDA_GENERIC
   ${MRAY_PREPROCESSOR_DEFS_GENERIC})

# External Libs
# add_subdirectory(Ext)

# BUILDS
# Libs
add_subdirectory(RayLib)

# DLLs
add_subdirectory(ImageIO)
add_subdirectory(VisorGL)

# Execs
add_subdirectory(MVisor)
