%module ellipsoid_cpp
%{
#include "ak_distance.h"
#include "histogram_merging.h"
#include "piecewise_linear.h"
#include "python_helpers.h"
%}

%include "std_pair.i"
%include "std_vector.i"

namespace std {
  %template(DoubleVector) vector<double>;
  %template(DoublePair) pair<double, double>;
  %template(IntPair) pair<int, int>;
  %template(DoublePairVector) vector<pair<double, double>>;
  %template(DoubleDoublePairVectorPair) pair<double, vector<pair<double, double>>>;
  %template(AkIntervalVector) vector<density_estimation::AkInterval>;
  %template(HistogramIntervalVector) vector<density_estimation::HistogramInterval>;
  %template(LinearPieceVector) vector<density_estimation::LinearPiece>;
}

%include "ak_distance.h"
%include "histogram_merging.h"
%include "piecewise_linear.h"
%include "python_helpers.h"
