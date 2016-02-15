CXX=g++-5
CXXFLAGS=-std=c++11 -DNDEBUG -Wall -Wextra -march=native -O3 -fPIC
SRCDIR=src
OBJDIR=obj

clean:
	rm -rf $(OBJDIR)
	rm -f _ellipsoid_cpp.so
	rm -f ellipsoid_cpp.py
	rm -f *.pyc
	rm -f *.gch

ellipsoid_swig: $(SRCDIR)/ak_distance.h $(SRCDIR)/python_helpers.h $(SRCDIR)/priority_queue.h $(SRCDIR)/ellipsoid_cpp.i
	swig -c++ -python -builtin -outcurrentdir $(SRCDIR)/ellipsoid_cpp.i
	mkdir -p $(OBJDIR)
	$(CXX) $(CXXFLAGS) `python-config --includes` -c ellipsoid_cpp_wrap.cxx -I src -o $(OBJDIR)/ellipsoid_cpp_wrap.o
	$(CXX) -shared $(OBJDIR)/ellipsoid_cpp_wrap.o -o _ellipsoid_cpp.so `python-config --ldflags` -lglpk
	rm -f ellipsoid_cpp_wrap.cxx
