cmake_minimum_required(VERSION 3.10)

project(oturin_a_1_gelu_omp VERSION 1.0.0 LANGUAGES CXX)

find_package(OpenMP REQUIRED)

file(GLOB HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)
file(GLOB SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB MAINFILE ${CMAKE_CURRENT_SOURCE_DIR}/*.cxx)

add_executable(${PROJECT_NAME} ${MAINFILE})

if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)
else()
    message(SEND_ERROR "OpenMP not found")
	return()
endif()

target_sources(${PROJECT_NAME} PRIVATE ${SOURCE})
target_include_directories(${PROJECT_NAME} PRIVATE ${HEADERS})
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_20)
target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -O3)
