#ifndef __AK_DISTANCE_H__
#define __AK_DISTANCE_H__

#include <cmath>
#include <vector>

#include "priority_queue.h"

namespace density_estimation {

struct Interval {
  double left;
  double right;
};

struct AkInterval {
  double value;
  double left;
  double right;
};


double eval_poly(const std::vector<double>& coeffs, double x) {
  double res = coeffs[0];
  for (size_t ii = 1; ii < coeffs.size(); ++ii) {
    res = res * x + coeffs[ii];
  }
  return res;
}


void negate_problem(std::vector<AkInterval>* l) {
  for (size_t ii = 0; ii < l->size(); ++ii) {
    (*l)[ii].value = -(((*l)[ii]).value);
  }
}


void construct_discrete_problem(const std::vector<double>& integral_coeffs,
                                const Interval& interval,
                                const std::vector<double>& samples,
                                double sample_weight,
                                std::vector<AkInterval>* result) {
  result->clear();
  result->reserve(2 * samples.size() + 1);
  double last_val = 0.0;
  double last_x = interval.left;

  for (int ii = 0; ii < static_cast<int>(samples.size()); ++ii) {
    double poly_val = eval_poly(integral_coeffs, samples[ii]);

    AkInterval cur;
    cur.left = last_x;
    cur.right = samples[ii];
    cur.value = poly_val - last_val;
    result->push_back(cur);

    cur.left = samples[ii];
    cur.right = samples[ii];
    cur.value = -sample_weight;
    result->push_back(cur);

    last_val = poly_val;
    last_x = samples[ii];
  }

  double total_val = eval_poly(integral_coeffs, interval.right);

  AkInterval last;
  last.left = last_x;
  last.right = interval.right;
  last.value = total_val - last_val;
  result->push_back(last);
}


bool solve_discrete_problem(const std::vector<AkInterval>& problem,
                            int k,
                            double* final_value,
                            std::vector<Interval>* result) {
  struct Part {
    double value;
    double left;
    double right;
    int next_left;
    int next_right;
  };
  const int kNoNeighbor = -1;

  if (k <= 0) {
    printf("k not positive\n");
    return false;
  }

  if (problem.size() <= 0) {
    printf("problem size 0\n");
    return false;
  }

  int leftmost_part = 0;
  int rightmost_part = problem.size() - 1;
  int num_nonnegative = 0;

  std::vector<Part> parts(problem.size());
  PriorityQueue<double, int> q(parts.size());
  for (size_t ii = 0; ii < parts.size(); ++ii) {
    parts[ii].value = problem[ii].value;
    parts[ii].left = problem[ii].left;
    parts[ii].right = problem[ii].right;
    if (ii == 0) {
      parts[ii].next_left = kNoNeighbor;
    } else {
      parts[ii].next_left = ii - 1;
    }
    if (ii + 1 == parts.size()) {
      parts[ii].next_right = kNoNeighbor;
    } else {
      parts[ii].next_right = ii + 1;
    }
    q.insert_unsorted(std::abs(parts[ii].value), ii);


    if (ii > 0 && ((parts[ii].value > 0 && parts[ii - 1].value > 0)
        || (parts[ii].value < 0 && parts[ii - 1].value < 0))) {
      printf("consecutive elements with the same sign:\n");
      printf("  ii = %lu\n", ii);
      printf("  parts[ii - 1].left = %lf\n", parts[ii - 1].left);
      printf("  parts[ii].right = %lf\n", parts[ii].right);
      printf("  parts[ii - 1].value = %lf\n", parts[ii - 1].value);
      printf("  parts[ii].value = %lf\n", parts[ii].value);
      return false;
    }

    if (parts[ii].value >= 0.0) {
      num_nonnegative += 1;
    }
  }

  q.make_heap();

  while (num_nonnegative > k) {
    if (parts[leftmost_part].value < 0.0) {
      //printf("deleting leftmost part %d\n", leftmost_part);
      q.delete_element(leftmost_part);
      leftmost_part = parts[leftmost_part].next_right;
      parts[leftmost_part].next_left = kNoNeighbor;
    }
    if (parts[rightmost_part].value < 0.0) {
      //printf("deleting rightmost part %d\n", rightmost_part);
      q.delete_element(rightmost_part);
      rightmost_part = parts[rightmost_part].next_left;
      parts[rightmost_part].next_right = kNoNeighbor;
    }

    int min_index;
    double min_value;
    q.get_min(&min_value, &min_index);
    //printf("current min_index: %d\n", min_index);

    if (parts[min_index].value >= 0.0) {
      num_nonnegative -= 1;
    }

    int neighbor_left = parts[min_index].next_left;
    int neighbor_right = parts[min_index].next_right;
    if (neighbor_left != kNoNeighbor) {
      //printf("deleting left neighbor %d\n", neighbor_left);
      q.delete_element(neighbor_left);
      if (parts[neighbor_left].value >= 0.0) {
        num_nonnegative -= 1;
      }
      parts[min_index].value += parts[neighbor_left].value;
      parts[min_index].left = parts[neighbor_left].left;
      parts[min_index].next_left = parts[neighbor_left].next_left;
      if (parts[min_index].next_left != kNoNeighbor) {
        parts[parts[min_index].next_left].next_right = min_index;
      }
      if (neighbor_left == leftmost_part) {
        leftmost_part = min_index;
      }
    }
    if (neighbor_right != kNoNeighbor) {
      //printf("deleting right neighbor %d\n", neighbor_right);
      q.delete_element(neighbor_right);
      if (parts[neighbor_right].value >= 0.0) {
        num_nonnegative -= 1;
      }
      parts[min_index].value += parts[neighbor_right].value;
      parts[min_index].right = parts[neighbor_right].right;
      parts[min_index].next_right = parts[neighbor_right].next_right;
      if (parts[min_index].next_right != kNoNeighbor) {
        parts[parts[min_index].next_right].next_left = min_index;
      }
      if (neighbor_right == rightmost_part) {
        rightmost_part = min_index;
      }
    }
    if (parts[min_index].value >= 0.0) {
      num_nonnegative += 1;
    }
    q.update_key(std::abs(parts[min_index].value), min_index);
  }

  result->clear();
  result->reserve(k);
  (*final_value) = 0.0;
  for (int ii = leftmost_part; ii != kNoNeighbor; ii = parts[ii].next_right) {
    if (parts[ii].value >= 0.0) {
      (*final_value) += parts[ii].value;
      Interval tmp;
      tmp.left = parts[ii].left;
      tmp.right = parts[ii].right;
      result->push_back(tmp);
    }
  }

  return true;
}


bool compute_ak(const std::vector<double>& integral_coeffs,
                const Interval& interval,
                const std::vector<double>& samples,
                double sample_weight,
                int k,
                double* value,
                std::vector<Interval>* result) {
  std::vector<AkInterval> problem;
  double val1;
  double val2;
  std::vector<Interval> sol1;
  std::vector<Interval> sol2;
  construct_discrete_problem(integral_coeffs, interval, samples, sample_weight,
      &problem);
  if (!solve_discrete_problem(problem, k, &val1, &sol1)) {
    return false;
  }
  negate_problem(&problem);
  if (!solve_discrete_problem(problem, k, &val2, &sol2)) {
    return false;
  }
  val2 = -val2;
  if (std::abs(val1) >= std::abs(val2)) {
    *value = val1;
    *result = sol1;
  } else {
    *value = val2;
    *result = sol2;
  }
  return true;
}



}  // namespace density_estimation

#endif
