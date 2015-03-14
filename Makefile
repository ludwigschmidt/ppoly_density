CXX=g++
CXXFLAGS=-std=c++11 -DNDEBUG -Wall -Wextra -march=native -O3 -fPIC
#INCLUDE_PATHS=~/Downloads/Clp-1.16.4/include/coin/ 
#LIBRARY_PATHS=~/Downloads/Clp-1.16.4/lib
INCLUDE_PATHS=""
LIBRARY_PATHS=""

ellipsoid_swig: ak_distance.h python_helpers.h priority_queue.h ellipsoid_cpp.i
	swig -c++ -python -builtin -outcurrentdir ellipsoid_cpp.i
	mkdir -p obj
	$(CXX) $(CXXFLAGS) `python-config --includes` -c ellipsoid_cpp_wrap.cxx -I src -o obj/ellipsoid_cpp_wrap.o
#	$(CXX) $(CXXFLAGS) `python-config --includes` -c ellipsoid_cpp_wrap.cxx -I src -I $(INCLUDE_PATHS) -o obj/ellipsoid_cpp_wrap.o
#	$(CXX) -shared obj/ellipsoid_cpp_wrap.o -o _ellipsoid_cpp.so `python-config --ldflags` -L $(LIBRARY_PATHS) -lClp -lClpSolver -lCoinUtils
	$(CXX) -shared obj/ellipsoid_cpp_wrap.o -o _ellipsoid_cpp.so `python-config --ldflags` -lglpk
	rm -f ellipsoid_cpp_wrap.cxx
