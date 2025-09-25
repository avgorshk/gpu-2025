cmake_minimum_required(VERSION 3.10)

project(oturin_a_1_gelu_omp VERSION 1.0.0 LANGUAGES CXX)

find_package(OpenMP REQUIRED)

file(GLOB HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)
file(GLOB SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB MAINFILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cxx)

foreach(MAINFILE ${MAINFILES})
    get_filename_component(EXECUTABLE_NAME ${MAINFILE} NAME_WE)
    set(EXEC_NAME "${PROJECT_NAME}_${EXECUTABLE_NAME}")

    add_executable(${EXEC_NAME} ${MAINFILE})

    if(OpenMP_CXX_FOUND)
        target_link_libraries(${EXEC_NAME} PUBLIC OpenMP::OpenMP_CXX)
    else()
        message(SEND_ERROR "OpenMP not found")
        return()
    endif()

    target_sources(${EXEC_NAME} PRIVATE ${SOURCE})
    target_include_directories(${EXEC_NAME} PRIVATE ${HEADERS})
    target_compile_features(${EXEC_NAME} PRIVATE cxx_std_20)
    target_compile_options(${EXEC_NAME} PRIVATE -Wall -Wextra -O3 -fopenmp-simd)
endforeach(MAINFILE ${MAINFILES})