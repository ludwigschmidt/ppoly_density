#ifndef __PYTHON_HELPERS_H__
#define __PYTHON_HELPERS_H__

#include <utility>
#include <vector>

#include "ak_distance.h"
#include "histogram_merging.h"

double solve_discrete_problem_cpp(
    const std::vector<density_estimation::AkInterval>& problem,
    int k,
    std::vector<std::pair<double, double>>* result) {
  result->clear();
  std::vector<density_estimation::Interval> tmp_result;
  double final_value;
  if (!density_estimation::solve_discrete_problem(problem, k, &final_value,
                                                  &tmp_result)) {
    return -1;
  } else {
    for (size_t ii = 0; ii < tmp_result.size(); ++ii) {
      result->push_back(std::make_pair(tmp_result[ii].left,
                                       tmp_result[ii].right));
    }
    return final_value;
  }
}

std::pair<double, std::vector<std::pair<double, double>>>
compute_ak_cpp(const std::vector<double>& integral_coeffs,
               std::pair<double, double> interval,
               const std::vector<double>& samples,
               double sample_weight,
               int k) {
  double result_val;
  std::vector<std::pair<double, double>> res;
  std::vector<density_estimation::Interval> result_sol;
  density_estimation::Interval int2;
  int2.left = interval.first;
  int2.right = interval.second;
  if (!compute_ak(integral_coeffs, int2, samples, sample_weight, k, &result_val,
      &result_sol)) {
    // TODO: better error handling
    return std::make_pair(-1.0, res);
  } else {
    for (size_t ii = 0; ii < result_sol.size(); ++ii) {
      res.push_back(std::make_pair(result_sol[ii].left, result_sol[ii].right));
    }
    return std::make_pair(result_val, res);
  }
}


std::pair<int, int>
histogram_merging_cpp(const std::vector<double>& samples,
                      double domain_left,
                      double domain_right,
                      int num_initial_intervals,
                      int num_merged_intervals_holdout,
                      int max_final_num_intervals,
                      std::vector<density_estimation::HistogramInterval>*
                          result) {
  int num_merging_iterations = 0;
  int num_a1_computations = 0;
  density_estimation::histogram_merging(samples, domain_left, domain_right,
      num_initial_intervals, num_merged_intervals_holdout,
      max_final_num_intervals, result, &num_merging_iterations,
      &num_a1_computations);
  return std::make_pair(num_merging_iterations, num_a1_computations);
}


#endif
