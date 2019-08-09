cmake_minimum_required(VERSION 3.6)

project(hello_tf)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if((CMAKE_CXX_COMPILER_ID MATCHES "GNU") OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
    set(CMAKE_VERBOSE_MAKEFILE ON)
    add_compile_options(-Wall -Wextra -pedantic-errors -Werror)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    add_compile_options(/W4 /WX)
    if(NOT CMAKE_CXX_COMPILER_VERSION VERSION_LESS 19.11)
        add_compile_options(/permissive-)
    endif()
endif()

if (NOT EXISTS ${CMAKE_SOURCE_DIR}/tensorflow)
    make_directory(${CMAKE_SOURCE_DIR}/tensorflow)
endif()

if(UNIX AND NOT APPLE)
    if (NOT EXISTS ${CMAKE_SOURCE_DIR}/tensorflow)
        make_directory(${CMAKE_SOURCE_DIR}/tensorflow)
    endif()

    include_directories(/usr/local/include/tensorflow/c)
    link_directories(/usr/local/lib)
endif()

include_directories(src/3rdparty/scope_guard)

add_executable(hello_tf src/hello_tf.cpp)
target_link_libraries(hello_tf tensorflow)

add_executable(session_run src/session_run.cpp src/tf_utils.cpp src/tf_utils.hpp)
target_link_libraries(session_run tensorflow)

add_executable(load_graph src/load_graph.cpp)
target_link_libraries(load_graph tensorflow)

add_executable(interface src/interface.cpp src/tf_utils.cpp src/tf_utils.hpp)
target_link_libraries(interface tensorflow)

add_executable(graph_info src/graph_info.cpp src/tf_utils.cpp src/tf_utils.hpp)
target_link_libraries(graph_info tensorflow)

add_executable(create_tensor src/create_tensor.cpp)
target_link_libraries(create_tensor tensorflow)

add_executable(tensor_info src/tensor_info.cpp src/tf_utils.cpp src/tf_utils.hpp)
target_link_libraries(tensor_info tensorflow)

add_executable(allocate_tensor src/allocate_tensor.cpp)
target_link_libraries(allocate_tensor tensorflow)

add_executable(batch_interface src/batch_interface.cpp src/tf_utils.cpp src/tf_utils.hpp)
target_link_libraries(batch_interface tensorflow)

configure_file(models/graph.pb ${CMAKE_CURRENT_BINARY_DIR}/graph.pb COPYONLY)

enable_testing()
add_subdirectory(test)
