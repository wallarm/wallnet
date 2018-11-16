#!/bin/bash
swig -c++ -python ./tandems/tandems.i
g++ -std=c++11 -fpic -c ./tandems/tandems.hpp ./tandems/tandems_wrap.cxx -I/usr/include/python3.6/
mv tandems_wrap.o ./tandems/tandems_wrap.o
gcc -shared ./tandems/tandems_wrap.o -o ./tandems/_tandems.so -lstdc++ -std=c++11