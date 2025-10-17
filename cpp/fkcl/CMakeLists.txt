# CMakeList.txt : CMake project for fkcl, include source and define
# project specific logic here.
#

# Add source to this project's executable.
add_executable (fkcl "fkcl.cpp" "fkcl.h" "kernels.h" "helpers.h")

target_include_directories(fkcl
    PRIVATE
        $ENV{OPENCL_DIR}/include
        $ENV{OPENCV_DIR}/../../include
        $ENV{BOOST_DIR}
)

target_link_directories(fkcl
    PRIVATE
        $ENV{OPENCL_DIR}/lib
        $ENV{OPENCV_DIR}/lib
)

target_link_libraries(fkcl
    PRIVATE
        OpenCL
        debug opencv_world4100d
        optimized opencv_world4100
)

file(GLOB RESOURCE_FILES "*.config")
set_source_files_properties(${RESOURCE_FILES} PROPERTIES HEADER_FILE_ONLY TRUE)

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET fkcl PROPERTY CXX_STANDARD 20)
endif()

