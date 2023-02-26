#!/bin/sh

mpiicpc -x c++ -std=c++17 -O3 -axCORE-AVX512 ./main.cpp
