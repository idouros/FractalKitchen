# CMakeList.txt : CMake project for fkcl, include source and define
# project specific logic here.
#

# Add source to this project's executable.
add_executable (fkcl "fkcl.cpp" "fkcl.h")

target_include_directories(fkcl
    PRIVATE
        $ENV{OPENCL_DIR}/include
)

target_link_directories(fkcl
    PRIVATE
        $ENV{OPENCL_DIR}/lib
)

target_link_libraries(fkcl
    PRIVATE
        OpenCL
)

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET fkcl PROPERTY CXX_STANDARD 20)
endif()

