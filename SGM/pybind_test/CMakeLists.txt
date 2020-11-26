cmake_minimum_required(VERSION 2.8.12)
project(main)

add_subdirectory(pybind11)
pybind11_add_module(main main.cpp)
