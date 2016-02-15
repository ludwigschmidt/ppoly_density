#ifndef __PIECEWISE_LINEAR_H__
#define __PIECEWISE_LINEAR_H__

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "ak_distance.h"
#include "lp_solver.h"

namespace density_estimation {

struct LinearPiece {
  double left;
  int left_sample_index;
  int right_sample_index;
  double right;
  double slope;
  double offset;
};

struct A1ProjectionStats {
  int num_merging_iterations;
  int num_a1_projections;
  int num_a1_computations;
  double weighted_num_a1_computations;
};

struct A1ProjectionOptions {
  int num_initial_interval_levels;
  double max_gap;
  int max_num_iterations;
};


double compute_a1_interval_error_slow(const LinearPiece& piece,
                                      const std::vector<double>& samples,
                                      double* opt_interval_left,
                                      double* opt_interval_right) {
  std::vector<double> coeffs(3);
  coeffs[0] = 0.5 * piece.slope;
  coeffs[1] = piece.offset;
  coeffs[2] = -0.5 * piece.slope * piece.left * piece.left
              - piece.offset * piece.left;
  Interval tmp_interval;
  tmp_interval.left = piece.left;
  tmp_interval.right = piece.right;
  int num_samples = piece.right_sample_index - piece.left_sample_index;
  std::vector<double> tmp_vector(num_samples);
  std::copy(samples.begin() + piece.left_sample_index,
            samples.begin() + piece.right_sample_index,
            tmp_vector.begin());
  double sample_weight = 1.0 / samples.size();
  int k = 1;
  double result;
  std::vector<Interval> tmp_result;
//  printf("%lf %lf %lf %lf %lf %lf %d %d\n ", coeffs[0], tmp_interval.left,
//     tmp_interval.right, tmp_vector[0], tmp_vector[num_samples - 1], sample_weight, k, num_samples);
  compute_ak(coeffs, tmp_interval, tmp_vector, sample_weight, k, &result,
      &tmp_result);
  *opt_interval_left = tmp_result[0].left;
  *opt_interval_right = tmp_result[0].right;
  return result;
}


int get_num_samples_exclusive(const std::vector<double>& samples, double left,
                              double right, int left_index, int right_index) {
  int left_search = left_index;
  if (samples[left_search] <= left) {
    int right_search_tmp = right_index;
    while (right_search_tmp - left_search > 1) {
      int mid = (left_search + right_search_tmp) / 2;
      if (samples[mid] <= left) {
        left_search = mid;
      } else {
        right_search_tmp = mid;
      }
    }
    left_search += 1;
  }

  int right_search = right_index - 1;
  if (left_search != right_index && samples[right_index - 1] >= right) {
    int left_search_tmp = left_search;
    while (right_search - left_search_tmp > 1) {
      int mid = (left_search_tmp + right_search) / 2;
      if (samples[mid] >= right) {
        right_search = mid;
      } else {
        left_search_tmp = mid;
      }
    }
    right_search -= 1;
  }

  if (left_search == right_search) {
    if (samples[left_search] <= left || samples[left_search] >= right) {
      right_search -= 1;
    }
  }

  /*int tmp = 0;
  for (int ii = left_index; ii < right_index; ++ii) {
    if (samples[ii] > left && samples[ii] < right) {
      tmp += 1;
    }
    printf("samples[%d] = %lf  tmp = %d\n", ii, samples[ii], tmp);
  }
  if (tmp != right_search - left_search + 1) {
    printf("ERROR: get_num_samples_exclusive consistency check failed.\n");
    printf("  left_search = %d  right_search = %d  expected difference = %d\n",
        left_search, right_search, tmp);
    printf("  left_index = %d  right_index = %d\n", left_index, right_index);
  }*/

  return right_search - left_search + 1;
}


int get_num_samples_inclusive(const std::vector<double>& samples, double left,
                              double right, int left_index, int right_index) {
  int left_search = left_index;
  if (samples[left_search] < left) {
    int right_search_tmp = right_index;
    while (right_search_tmp - left_search > 1) {
      int mid = (left_search + right_search_tmp) / 2;
      if (samples[mid] < left) {
        left_search = mid;
      } else {
        right_search_tmp = mid;
      }
    }
    left_search += 1;
  }

  int right_search = right_index - 1;
  if (left_search != right_index && samples[right_index - 1] > right) {
    int left_search_tmp = left_search;
    while (right_search - left_search_tmp > 1) {
      int mid = (left_search_tmp + right_search) / 2;
      if (samples[mid] > right) {
        right_search = mid;
      } else {
        left_search_tmp = mid;
      }
    }
    right_search -= 1;
  }

  /*int tmp = 0;
  for (int ii = left_index; ii < right_index; ++ii) {
    if (samples[ii] >= left && samples[ii] <= right) {
      tmp += 1;
    }
    //printf("samples[%d] = %lf  tmp = %d\n", ii, samples[ii], tmp);
  }
  if (tmp != right_search - left_search + 1) {
    printf("ERROR: get_num_samples_inclusive consistency check failed.\n");
    printf("  left_search = %d  right_search = %d  expected difference = %d\n",
        left_search, right_search, tmp);
    printf("  left_index = %d  right_index = %d\n", left_index, right_index);
  }*/

  return right_search - left_search + 1;
}


double integrate(const LinearPiece& piece, double from, double to) {
  return .5 * piece.slope * (to*to - from*from) + piece.offset * (to - from);
}

double compute_a1_interval_error(const LinearPiece& piece,
                                 const std::vector<double>& samples,
                                 double* opt_interval_left,
                                 double* opt_interval_right) {
  double best = 0.0;
  double sample_weight = 1.0 / samples.size();

  // Pass 1: positive intervals
  double opt_pos_interval_left = piece.left;
  double opt_pos_interval_right = samples[piece.left_sample_index];
  int cur_index = piece.left_sample_index;
  int cur_start = cur_index;
  double cur_sum = integrate(piece, piece.left, samples[cur_index]);
  best = cur_sum;

  while (cur_index != piece.right_sample_index - 1) {
    cur_index += 1;
    if (cur_sum <= sample_weight) {
      cur_start = cur_index;
      cur_sum = 0.0;
    } else {
      cur_sum -= sample_weight;
    }
    cur_sum += integrate(piece, samples[cur_index - 1], samples[cur_index]);
    if (cur_sum > best) {
      best = cur_sum;
//      printf("pos case 1\n");
//      printf("  cur_start = %d  cur_index = %d\n", cur_start, cur_index);
//      printf("  %lf %lf %lf\n", samples[cur_start - 2], samples[cur_start - 1], samples[cur_start]);
//      printf("  %lf %lf %lf\n", samples[cur_index - 1], samples[cur_index], samples[cur_index + 1]);
      opt_pos_interval_left = (cur_start == piece.left_sample_index ?
                               piece.left :
                               samples[cur_start - 1]);
      opt_pos_interval_right = samples[cur_index];
    }
  }
  cur_index += 1;
  if (cur_sum <= sample_weight) {
    cur_start = cur_index;
    cur_sum = 0.0;
  } else {
    cur_sum -= sample_weight;
  }
  cur_sum += integrate(piece, samples[cur_index - 1], piece.right);
  if (cur_sum > best) {
      best = cur_sum;
//      printf("pos case 2\n");
      opt_pos_interval_left = (cur_start == piece.left_sample_index ?
                               piece.left :
                               samples[cur_start - 1]);
      opt_pos_interval_right = piece.right;
  }

  int left_start = cur_start;
  double left_sum = cur_sum;
  cur_index = piece.right_sample_index;
  cur_start = cur_index;
  cur_sum = integrate(piece, samples[cur_index - 1], piece.right);
  while (cur_index > left_start) {
    cur_index -= 1;
    if (cur_sum <= sample_weight) {
      left_sum -= cur_sum;
      left_sum += sample_weight;
      if (left_sum > best) {
        best = left_sum;
//        printf("pos case 3\n");
        opt_pos_interval_left = (left_start == piece.left_sample_index ?
                                 piece.left :
                                 samples[left_start - 1]);
        opt_pos_interval_right = samples[cur_index];
      }
      cur_sum = 0.0;
      cur_start = cur_index;
    } else {
      cur_sum -= sample_weight;
    }
    if (cur_index != piece.left_sample_index) {
      cur_sum += integrate(piece, samples[cur_index - 1], samples[cur_index]);
    } else {
      cur_sum += integrate(piece, piece.left, samples[cur_index]);
    }
    if (cur_sum > best) {
      best = cur_sum;
//      printf("pos case 4\n");
      opt_pos_interval_left = (cur_index == piece.left_sample_index ?
                               piece.left :
                               samples[cur_index - 1]);
      opt_pos_interval_right = (cur_start == piece.right_sample_index ?
                               piece.right :
                               samples[cur_start]);
    }
  }


  // Pass 2: negative intervals
  double opt_neg_interval_left = samples[piece.left_sample_index];
  double opt_neg_interval_right = samples[piece.left_sample_index];
  double best_neg = 0.0;
  cur_index = piece.left_sample_index;
  cur_start = cur_index;
  cur_sum = -sample_weight;
  best_neg = cur_sum;

  while (cur_index != piece.right_sample_index - 1) {
    cur_index += 1;
    double piece_val = integrate(piece, samples[cur_index - 1],
        samples[cur_index]);
    if (-cur_sum <= piece_val) {
      cur_start = cur_index;
      cur_sum = 0.0;
    } else {
      cur_sum += piece_val;
    }
    cur_sum -= sample_weight;
    if (cur_sum < best_neg) {
      best_neg = cur_sum;
      opt_neg_interval_left = samples[cur_start];
      opt_neg_interval_right = samples[cur_index];
//      printf("neg case 1, num_samples = %d\n", cur_index - cur_start + 1);
//      printf("  cur_start = %d  cur_index = %d\n", cur_start, cur_index);
    }
  }

  left_start = cur_start;
  left_sum = cur_sum;
  cur_index = piece.right_sample_index - 1;
  cur_start = cur_index;
  cur_sum = -sample_weight;

//  if (std::abs(-left_sum - sample_weight * (interval.right_sample_index - left_start) + val * (samples[interval.right_sample_index - 1] - samples[left_start])) > 0.000001) {
//    printf("-------- INITIAL CONSISTENCY CHECK FAILED\n");
//  } else {
//    printf("-------- INITIAL CONSISTENCY CHECK PASSED\n");
//  }

  while (cur_index > left_start) {
    cur_index -= 1;
    double piece_val = integrate(piece, samples[cur_index],
        samples[cur_index + 1]);
    if (-cur_sum <= piece_val) {
      left_sum -= cur_sum;
      left_sum -= piece_val;
//      printf("Negative right discarding\n");
//      if (std::abs(-left_sum + -sample_weight * (cur_index - left_start + 1)
//          + val * (samples[cur_index] - samples[left_start])) > 0.000001) {
//        printf("CONSISTENCY CHECK FAILED\nleft_start = %d\ncur_index = %d\nleft_sample_index = %d\nright_sample_index - 1 = %d\nleft_sum = %lf\ncur_sum = %lf\nhist_val = %lf\nexpected_value = %lf\n", left_start, cur_index, interval.left_sample_index, interval.right_sample_index - 1, left_sum, cur_sum, hist_val, -sample_weight * (cur_index - left_start + 1) + val * (samples[cur_index] - samples[left_start]));
//      }
      if (left_sum < best_neg) {
        best_neg = left_sum;
//       printf("neg case 2\n");
        opt_neg_interval_left = samples[left_start];
        opt_neg_interval_right = samples[cur_index];
      }
      cur_sum = 0.0;
      cur_start = cur_index;
    } else {
      cur_sum += piece_val;
    }
    cur_sum -= sample_weight;
    best_neg = std::min(best_neg, cur_sum);
    if (cur_sum < best_neg) {
      best_neg = cur_sum;
//      printf("neg case 3\n");
      opt_neg_interval_left = samples[cur_index];
      opt_neg_interval_right = samples[cur_start];
    }
  }

  if (std::abs(best) > std::abs(best_neg)) {
    *opt_interval_left = opt_pos_interval_left;
    *opt_interval_right = opt_pos_interval_right;

    /*
    //////
    double tmpl, tmpr;
    double tmpv = compute_a1_interval_error_slow(piece, samples, &tmpl, &tmpr);
    double linmass = integrate(piece, *opt_interval_left, *opt_interval_right);
    double samplemass = get_num_samples_exclusive(samples, *opt_interval_left,
              *opt_interval_right, piece.left_sample_index,
              piece.right_sample_index) * sample_weight;

    if (std::abs(best - std::abs(tmpv)) > 0.000001
        || std::abs(linmass - samplemass - best) > 0.000001) {
      printf("ERROR: pos A1 consistency check failed.\n");
      printf("slope = %lf  offset = %lf\n", piece.slope, piece.offset);
      printf("left_sample_index = %d  right_sample_index = %d\n",
          piece.left_sample_index, piece.right_sample_index);
      printf("A1 = %lf  expected = %lf\n", best, tmpv);
      printf("left = %lf  right = %lf\n", *opt_interval_left,
          *opt_interval_right);
      printf("linmass = %lf  samplemass = %lf\n", linmass, samplemass);
    } //else {
//      printf("SUCCESS: pos A1 consistency check passed.\n");
//    }
    */
    
    return best;
  } else {
    *opt_interval_left = opt_neg_interval_left;
    *opt_interval_right = opt_neg_interval_right;

    /*
    //////
    double tmpl, tmpr;
    double tmpv = compute_a1_interval_error_slow(piece, samples, &tmpl, &tmpr);
    double linmass = integrate(piece, *opt_interval_left, *opt_interval_right);
    double samplemass = get_num_samples_inclusive(samples, *opt_interval_left,
              *opt_interval_right, piece.left_sample_index,
              piece.right_sample_index) * sample_weight;

    if (std::abs(best_neg + std::abs(tmpv)) > 0.000001
        || std::abs(linmass - samplemass - best_neg) > 0.000001) {
      printf("ERROR: neg A1 consistency check failed.\n");
      printf("slope = %lf  offset = %lf\n", piece.slope, piece.offset);
      printf("left_sample_index = %d  right_sample_index = %d\n",
          piece.left_sample_index, piece.right_sample_index);
      printf("A1 = %lf  expected = %lf\n", best_neg, tmpv);
      printf("left = %lf  right = %lf\n", *opt_interval_left,
          *opt_interval_right);
      printf("linmass = %lf  samplemass = %lf\n", linmass, samplemass);
    } //else {
//      printf("SUCCESS: neg A1 consistency check passed.\n");
//    }
    */

    return best_neg;
  }
}


double compute_a1_interval_error2(const LinearPiece& piece,
                                  const std::vector<double>& samples,
                                  double* opt_interval_left,
                                  double* opt_interval_right) {
  double sample_weight = 1.0 / samples.size();
  int cur_index = piece.left_sample_index - 1;
  double cur_x = piece.left;
  int largest_index = cur_index;
  double largest = piece.offset * cur_x + 0.5 * piece.slope * cur_x * cur_x;
  int smallest_index = cur_index;
  double smallest = largest;
  double sum = largest;

  while (cur_index != piece.right_sample_index - 1) {
    cur_index += 1;
    cur_x = samples[cur_index];
    sum = piece.offset * cur_x + 0.5 * piece.slope * cur_x * cur_x
        - (cur_index - piece.left_sample_index) * sample_weight;
    if (sum > largest) {
      largest = sum;
      largest_index = cur_index;
    }
    sum -= sample_weight;
    if (sum < smallest) {
      smallest = sum;
      smallest_index = cur_index;
    }
  }

  cur_x = piece.right;
  sum = piece.offset * cur_x + 0.5 * piece.slope * cur_x * cur_x
      - (piece.right_sample_index - piece.left_sample_index) * sample_weight;
  if (sum > largest) {
    largest = sum;
    largest_index = piece.right_sample_index;
  }
  if (sum < smallest) {
    smallest = sum;
    smallest_index = piece.right_sample_index;
  }

  double best = 0.0;

  if (largest_index > smallest_index) {
    best = largest - smallest;
    *opt_interval_left = (smallest_index == piece.left_sample_index - 1 ?
                          piece.left : samples[smallest_index]);
    *opt_interval_right = (largest_index == piece.right_sample_index ?
                           piece.right : samples[largest_index]);
    ///////
    /*
    double tmpl, tmpr;
    double tmpv = compute_a1_interval_error_slow(piece, samples, &tmpl, &tmpr);
    double linmass = piece.offset * *opt_interval_right
                     + 0.5 * piece.slope * *opt_interval_right
                        * *opt_interval_right
                     - piece.offset * *opt_interval_left
                     - 0.5 * piece.slope * *opt_interval_left
                        * *opt_interval_left;
    double samplemass = get_num_samples_exclusive(samples, *opt_interval_left,
              *opt_interval_right, piece.left_sample_index,
              piece.right_sample_index) * sample_weight;
    if (std::abs(best - std::abs(tmpv)) > 0.000001
        || std::abs(linmass - samplemass - best) > 0.000001) {
      printf("ERROR: Consistency check failed in pos part of compute a1.\n");
      printf("  slope = %lf  offset = %lf\n", piece.slope, piece.offset);
      printf("  left_sample_index = %d  right_sample_index = %d\n",
          piece.left_sample_index, piece.right_sample_index);
      printf("  A1 = %lf  expected = %lf\n", best, tmpv);
      printf("  left = %lf  right = %lf\n", *opt_interval_left,
          *opt_interval_right);
      printf("  linmass = %lf  samplemass = %lf\n", linmass, samplemass);
      printf("  largest_index = %d  smallest_index = %d\n", largest_index, smallest_index);
      printf("  largest = %lf  smallest = %lf\n", largest, smallest);
    } else {
      printf("SUCCESS: consistency check passed in pos part of compute a1.\n");
    }
    */

    return best;
  } else {
    best = -(largest - smallest);
    *opt_interval_left = (largest_index == piece.left_sample_index - 1 ?
                          piece.left : samples[largest_index]);
    *opt_interval_right = (smallest_index == piece.right_sample_index ?
                          piece.right : samples[smallest_index]);

    ///////
    /*
    double tmpl, tmpr;
    double tmpv = compute_a1_interval_error_slow(piece, samples, &tmpl, &tmpr);
    double linmass = piece.offset * *opt_interval_right
                     + 0.5 * piece.slope * *opt_interval_right
                        * *opt_interval_right
                     - piece.offset * *opt_interval_left
                     - 0.5 * piece.slope * *opt_interval_left
                        * *opt_interval_left;
    double samplemass = get_num_samples_inclusive(samples, *opt_interval_left,
              *opt_interval_right, piece.left_sample_index,
              piece.right_sample_index) * sample_weight;
    if (std::abs(-best - std::abs(tmpv)) > 0.000001
        || std::abs(linmass - samplemass - best) > 0.000001) {
      printf("ERROR: Consistency check failed in neg part of compute a1.\n");
      printf("  slope = %lf  offset = %lf\n", piece.slope, piece.offset);
      printf("  left_sample_index = %d  right_sample_index = %d\n",
          piece.left_sample_index, piece.right_sample_index);
      printf("  A1 = %lf  expected = %lf\n", best, tmpv);
      printf("  left = %lf  right = %lf\n", *opt_interval_left,
          *opt_interval_right);
      printf("  linmass = %lf  samplemass = %lf\n", linmass, samplemass);
      printf("  largest_index = %d  smallest_index = %d\n", largest_index, smallest_index);
      printf("  largest = %lf  smallest = %lf\n", largest, smallest);
    } else {
      printf("SUCCESS: consistency check passed in neg part of compute a1.\n");
    }
    */

    return -(largest - smallest);
  }
}




class A1ProjectionLinear {
 public:
  A1ProjectionLinear(const A1ProjectionOptions& opts) : opts_(opts), solver_(3) {
    coeffs[0] = 0.0;
    coeffs[1] = 0.0;
    coeffs[2] = 1.0;
    solver_.set_objective(coeffs);
  }

  double project(LinearPiece* piece,
                 const std::vector<double>& samples,
                 int* num_a1_computations) {
    *num_a1_computations = 0;
    double sample_weight = 1.0 / samples.size();
    double total_weight = sample_weight *
        (piece->right_sample_index - piece->left_sample_index);
    double a = piece->left;
    double b = piece->right;

    // Early exit if the total weight is smaller than the desired precision.
    // In that case, the flat function achieves OPT up to the desired precision.
    if (total_weight < opts_.max_gap) {
      piece->slope = 0.0;
      piece->offset = 0.0;
      return total_weight;
    }

    solver_.reset_constraints();
    coeffs[0] = -a;
    coeffs[1] = -1.0;
    coeffs[2] = 0.0;
    solver_.add_constraint(coeffs, 0);
    coeffs[0] = -b;
    coeffs[1] = -1.0;
    coeffs[2] = 0.0;
    solver_.add_constraint(coeffs, 0);
    coeffs[0] = (b*b - a*a) / 2.0;
    coeffs[1] = b - a;
    coeffs[2] = -1.0;
    solver_.add_constraint(coeffs, total_weight);
    coeffs[0] = (a*a - b*b) / 2.0;
    coeffs[1] = a - b;
    coeffs[2] = -1.0;
    solver_.add_constraint(coeffs, -total_weight);

    int num_hierarchy_parts = 1;
    for (int ii = 0; ii < opts_.num_initial_interval_levels - 1; ++ii) {
      num_hierarchy_parts *= 2;
      double interval_width = (b - a) / num_hierarchy_parts;

      for (int jj = 0; jj < num_hierarchy_parts; ++jj) {
        double c = a + jj * interval_width;
        double d = c + interval_width;
        double wpos = sample_weight * get_num_samples_exclusive(samples, c, d,
            piece->left_sample_index, piece->right_sample_index);
        double wneg = sample_weight * get_num_samples_inclusive(samples, c, d,
            piece->left_sample_index, piece->right_sample_index);
        coeffs[0] = (d*d - c*c) / 2.0;
        coeffs[1] = d - c;
        coeffs[2] = -1.0;
        solver_.add_constraint(coeffs, wpos);
//        printf("Adding constraint for (%lf, %lf), weight = %lf\n", c, d, wpos);
        coeffs[0] = (c*c - d*d) / 2.0;
        coeffs[1] = c - d;
        coeffs[2] = -1.0;
        solver_.add_constraint(coeffs, -wneg);
//        printf("Adding constraint for (%lf, %lf), weight = %lf\n", c, d, -wneg);
      }
      
      for (int jj = 0; jj < num_hierarchy_parts - 1; ++jj) {
        double c = a + jj * interval_width + interval_width / 2.0;
        double d = c + interval_width;
        double wpos = sample_weight * get_num_samples_exclusive(samples, c, d,
            piece->left_sample_index, piece->right_sample_index);
        double wneg = sample_weight * get_num_samples_inclusive(samples, c, d,
            piece->left_sample_index, piece->right_sample_index);
        coeffs[0] = (d*d - c*c) / 2.0;
        coeffs[1] = d - c;
        coeffs[2] = -1.0;
        solver_.add_constraint(coeffs, wpos);
//        printf("Adding constraint for (%lf, %lf), weight = %lf\n", c, d, wpos);
        coeffs[0] = (c*c - d*d) / 2.0;
        coeffs[1] = c - d;
        coeffs[2] = -1.0;
        solver_.add_constraint(coeffs, -wneg);
//        printf("Adding constraint for (%lf, %lf), weight = %lf\n", c, d, -wneg);
      }
    }

    double lower_bound = 0.0;
    double best = total_weight;
    double best_slope = 0.0;
    double best_offset = 0.0;
    int num_iter = 0;

    while (best - lower_bound > opts_.max_gap
           && num_iter <= opts_.max_num_iterations) {
      num_iter += 1;

      if (!solver_.solve(coeffs)) {
        printf("ERROR RETURNED BY SOLVER\n");
        // TODO: handle error
      }

      lower_bound = coeffs[2];
      piece->slope = coeffs[0];
      piece->offset = coeffs[1];

      double c, d;
      double a1_dst = compute_a1_interval_error2(*piece, samples, &c, &d);
      *num_a1_computations += 1;
      
//      printf("ii = %d  a1_dst = %lf  best = %lf  lower_bound = %lf\n",
//          num_iter, a1_dst, best, lower_bound);
//      printf("  slope = %lf  offset = %lf\n", piece->slope, piece->offset);
      
      if (std::abs(a1_dst) < best) {
        best_slope = piece->slope;
        best_offset = piece->offset;
        best = std::abs(a1_dst);
//        printf("  new best = %lf\n", best);
      }

      double samples_weight = integrate(*piece, c, d) - a1_dst;

      if (a1_dst >= 0.0) {
        coeffs[0] = (d*d - c*c) / 2.0;
        coeffs[1] = d - c;
        coeffs[2] = -1.0;
        solver_.add_constraint(coeffs, samples_weight);
//        printf("Adding constraint for (%lf, %lf), weight = %lf\n", c, d, samples_weight);
      } else {
        coeffs[0] = (c*c - d*d) / 2.0;
        coeffs[1] = c - d;
        coeffs[2] = -1.0;
        solver_.add_constraint(coeffs, -samples_weight);
//        printf("Adding constraint for (%lf, %lf), weight = %lf\n", c, d, -samples_weight);
      }
    }

    piece->slope = best_slope;
    piece->offset = best_offset;
    return best;
  }

 private:
  A1ProjectionOptions opts_;
  LPSolver solver_;
  double coeffs[3];
};


bool piecewise_linear_approx(const std::vector<double>& samples,
                             double domain_left,
                             double domain_right,
                             int num_initial_pieces,
                             int num_merged_pieces_holdout,
                             int max_final_num_pieces,
                             const A1ProjectionOptions& projection_opts,
                             std::vector<LinearPiece>* result,
                             A1ProjectionStats* stats) {
  A1ProjectionLinear proj(projection_opts);
  
  // Step 0: initialization
  stats->num_merging_iterations = 0;
  stats->num_a1_computations = 0;
  stats->weighted_num_a1_computations = 0;
  stats->num_merging_iterations = 0;
  std::vector<LinearPiece> v1(num_initial_pieces);
  std::vector<LinearPiece> v2(num_initial_pieces);
  std::vector<LinearPiece> candidate_pieces(num_initial_pieces);
  std::vector<std::pair<double, int>> errors1(num_initial_pieces);
  std::vector<std::pair<double, int>> errors2(num_initial_pieces);
  int n = samples.size();
  int cur_num_a1_computations = 0;

  // Step 1: build initial pieces
  int samples_per_piece = n / num_initial_pieces;
  int num_extra_samples = n - samples_per_piece * num_initial_pieces;
  double last_boundary = domain_left;
  int last_sample_index = 0;

  for (int ii = 0; ii < num_initial_pieces; ++ii) {
    LinearPiece& cur_piece= v1[ii];
    cur_piece.left = last_boundary;
    cur_piece.left_sample_index = last_sample_index;
    cur_piece.right_sample_index = last_sample_index + samples_per_piece;
    if (ii < num_extra_samples) {
      cur_piece.right_sample_index += 1;
    }
    if (ii == num_initial_pieces - 1) {
      cur_piece.right = domain_right;
    } else {
      cur_piece.right = (samples[cur_piece.right_sample_index - 1]
                            + samples[cur_piece.right_sample_index]) / 2.0;
    }
    proj.project(&cur_piece, samples, &cur_num_a1_computations);
    stats->num_a1_computations += cur_num_a1_computations;
    stats->weighted_num_a1_computations += cur_num_a1_computations *
        (cur_piece.right_sample_index - cur_piece.left_sample_index) /
        static_cast<double>(n);
    stats->num_a1_projections += 1;
    last_boundary = cur_piece.right;
    last_sample_index = cur_piece.right_sample_index;
  }

  // Step 2: merging
  int cur_num_pieces = num_initial_pieces;
  int prev_num_pieces= 0;
  std::vector<LinearPiece>* cur_pieces_p = &v1;
  std::vector<LinearPiece>* prev_pieces_p = &v2;
  std::vector<LinearPiece>* tmp_pieces_p;
  
  while (cur_num_pieces > max_final_num_pieces) {
    //printf("\ncur_num_pieces: %d\n", cur_num_pieces);

    // Step 2.0: update references
    stats->num_merging_iterations += 1;
    prev_num_pieces = cur_num_pieces;
    tmp_pieces_p = prev_pieces_p;
    prev_pieces_p = cur_pieces_p;
    cur_pieces_p = tmp_pieces_p;
    std::vector<LinearPiece>& cur_pieces = *cur_pieces_p;
    std::vector<LinearPiece>& prev_pieces = *prev_pieces_p;
    
    // Step 2.1: computing errors
    int num_candidates = prev_num_pieces / 2;
    for (int ii = 0; ii < num_candidates; ++ii) {
      candidate_pieces[ii].left = prev_pieces[2 * ii].left;
      candidate_pieces[ii].right = prev_pieces[2 * ii + 1].right;
      candidate_pieces[ii].left_sample_index =
          prev_pieces[2 * ii].left_sample_index;
      candidate_pieces[ii].right_sample_index =
          prev_pieces[2 * ii + 1].right_sample_index;
      double err = proj.project(&(candidate_pieces[ii]),
                                samples,
                                &cur_num_a1_computations);
      
      
      /*double err2 = compute_a1_interval_error_slow(candidate_pieces[ii], samples);
      if (std::abs(err - err2) > 0.0000001) {
        printf("ERROR: A1 distance incorrect for ii = %d. Result %lf but should be %lf\n", ii, err, err2);
      }*/


      errors1[ii] = std::make_pair(err, ii);
      stats->num_a1_projections += 1;
      stats->num_a1_computations += cur_num_a1_computations;
      stats->weighted_num_a1_computations += cur_num_a1_computations *
          (candidate_pieces[ii].right_sample_index
              - candidate_pieces[ii].left_sample_index) / static_cast<double>(n);
    }

    // Step 2.2: find threshold
    std::copy(errors1.begin(),
              errors1.begin() + num_candidates,
              errors2.begin());
    int threshold_pos = num_candidates - num_merged_pieces_holdout;
    std::nth_element(errors2.begin(),
                     errors2.begin() + threshold_pos,
                     errors2.begin() + num_candidates);
    const std::pair<double, int>& error_threshold = errors2[threshold_pos];
   
    // Step 2.3: build new hypothesis 
    cur_num_pieces = 0;
    for (int ii = 0; ii < num_candidates; ++ii) {
      if (errors1[ii] >= error_threshold) {
        cur_pieces[cur_num_pieces] = prev_pieces[2 * ii];
        cur_pieces[cur_num_pieces + 1] = prev_pieces[2 * ii + 1];
        cur_num_pieces += 2;
      } else {
        cur_pieces[cur_num_pieces] = candidate_pieces[ii];
        cur_num_pieces += 1;
      }
    }
    if (prev_num_pieces % 2 == 1) {
      cur_pieces[cur_num_pieces] = prev_pieces[prev_num_pieces - 1];
      cur_num_pieces += 1;
    }
  }

  // Step 3: return result
  result->resize(cur_num_pieces);
  std::copy(cur_pieces_p->begin(),
            cur_pieces_p->begin() + cur_num_pieces,
            result->begin());

  return true;
}


}  // namespace density_estimation


#endif
