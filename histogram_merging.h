#ifndef __HISTOGRAM_MERGING_H__
#define __HISTOGRAM_MERGING_H__

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "ak_distance.h"

namespace density_estimation {

struct HistogramInterval {
  double left;
  int left_sample_index;
  int right_sample_index;
  double right;
  double value;
};


double compute_interval_hypothesis(const HistogramInterval& interval,
                                   double sample_weight) {
/*  if (interval.right_sample_index < interval.left_sample_index) {
    printf("Error1: %lf %lf %d %d\n", interval.left, interval.right,
        interval.left_sample_index, interval.right_sample_index);
  }
  if (interval.right < interval.left) {
    printf("Error2: %lf %lf %d %d\n", interval.left, interval.right,
        interval.left_sample_index, interval.right_sample_index);
  }*/
  return (interval.right_sample_index - interval.left_sample_index)
         * sample_weight / (interval.right - interval.left);
}


/*double compute_a1_interval_error_slow(const HistogramInterval& interval,
                                      const std::vector<double>& samples) {
  std::vector<double> coeffs(2);
  coeffs[0] = interval.value;
  coeffs[1] = -interval.left * interval.value;
  Interval tmp_interval;
  tmp_interval.left = interval.left;
  tmp_interval.right = interval.right;
  int num_samples = interval.right_sample_index - interval.left_sample_index;
  std::vector<double> tmp_vector(num_samples);
  std::copy(samples.begin() + interval.left_sample_index,
            samples.begin() + interval.right_sample_index,
            tmp_vector.begin());
  double sample_weight = 1.0 / samples.size();
  int k = 1;
  double result;
  std::vector<Interval> tmp_result;
//  printf("%lf %lf %lf %lf %lf %lf %d %d\n ", coeffs[0], tmp_interval.left,
//     tmp_interval.right, tmp_vector[0], tmp_vector[num_samples - 1], sample_weight, k, num_samples);
  compute_ak(coeffs, tmp_interval, tmp_vector, sample_weight, k, &result,
      &tmp_result);
  return std::abs(result);
}*/


double compute_a1_interval_error(const HistogramInterval& interval,
                                 const std::vector<double>& samples) {
  double val = interval.value;
  double sample_weight = 1.0 / samples.size();
  int cur_index = interval.left_sample_index;
  double largest = val * (samples[cur_index] - interval.left);
  double smallest = std::min(0.0, largest - sample_weight);
  double sum = largest - sample_weight;

  while (cur_index != interval.right_sample_index - 1) {
    cur_index += 1;
    sum += val * (samples[cur_index] - samples[cur_index - 1]);
    largest = std::max(largest, sum);
    sum -= sample_weight;
    smallest = std::min(smallest, sum);
  }

  sum += val * (interval.right - samples[interval.right_sample_index - 1]);
  largest = std::max(largest, sum);

  return std::abs(largest - smallest);
}


/*double compute_a1_interval_error2(const HistogramInterval& interval,
                                  const std::vector<double>& samples) {
  double best = 0.0;
  double val = interval.value;
  double sample_weight = 1.0 / samples.size();

  // Pass 1: positive intervals
  int cur_index = interval.left_sample_index;
  int cur_start = cur_index;
  double cur_sum = val * (samples[cur_index] - interval.left);
  best = cur_sum;

  while (cur_index != interval.right_sample_index - 1) {
    cur_index += 1;
    if (cur_sum <= sample_weight) {
      cur_start = cur_index;
      cur_sum = 0.0;
    } else {
      cur_sum -= sample_weight;
    }
    cur_sum += val * (samples[cur_index] - samples[cur_index - 1]);
    best = std::max(best, cur_sum);
  }
  cur_index += 1;
  if (cur_sum <= sample_weight) {
    cur_start = cur_index;
    cur_sum = 0.0;
  } else {
    cur_sum -= sample_weight;
  }
  cur_sum += val * (interval.right - samples[cur_index - 1]);
  best = std::max(best, cur_sum);

  int left_start = cur_start;
  double left_sum = cur_sum;
  cur_index = interval.right_sample_index;
  cur_start = cur_index;
  cur_sum = val * (interval.right - samples[interval.right_sample_index - 1]);
  while (cur_index > left_start) {
    cur_index -= 1;
    if (cur_sum <= sample_weight) {
      left_sum -= cur_sum;
      left_sum += sample_weight;
      best = std::max(best, left_sum);
      cur_sum = 0.0;
      cur_start = cur_index;
    } else {
      cur_sum -= sample_weight;
    }
    cur_sum += val * (samples[cur_index] - samples[cur_index - 1]);
    best = std::max(best, cur_sum);
  }


  // Pass 2: negative intervals
  double best_neg = 0.0;
  cur_index = interval.left_sample_index;
  cur_start = cur_index;
  cur_sum = -sample_weight;
  best_neg = cur_sum;

  while (cur_index != interval.right_sample_index - 1) {
    cur_index += 1;
    double hist_val = val * (samples[cur_index] - samples[cur_index - 1]);
    if (-cur_sum <= hist_val) {
      cur_start = cur_index;
      cur_sum = 0.0;
    } else {
      cur_sum += hist_val;
    }
    cur_sum -= sample_weight;
    best_neg = std::min(best_neg, cur_sum);
  }

  left_start = cur_start;
  left_sum = cur_sum;
  cur_index = interval.right_sample_index - 1;
  cur_start = cur_index;
  cur_sum = -sample_weight;

//  if (std::abs(-left_sum - sample_weight * (interval.right_sample_index - left_start) + val * (samples[interval.right_sample_index - 1] - samples[left_start])) > 0.000001) {
//    printf("-------- INITIAL CONSISTENCY CHECK FAILED\n");
//  } else {
//    printf("-------- INITIAL CONSISTENCY CHECK PASSED\n");
//  }

  while (cur_index > left_start) {
    cur_index -= 1;
    double hist_val = val * (samples[cur_index + 1] - samples[cur_index]);
    if (-cur_sum <= hist_val) {
      left_sum -= cur_sum;
      left_sum -= hist_val;
//      printf("Negative right discarding\n");
//      if (std::abs(-left_sum + -sample_weight * (cur_index - left_start + 1)
//          + val * (samples[cur_index] - samples[left_start])) > 0.000001) {
//        printf("CONSISTENCY CHECK FAILED\nleft_start = %d\ncur_index = %d\nleft_sample_index = %d\nright_sample_index - 1 = %d\nleft_sum = %lf\ncur_sum = %lf\nhist_val = %lf\nexpected_value = %lf\n", left_start, cur_index, interval.left_sample_index, interval.right_sample_index - 1, left_sum, cur_sum, hist_val, -sample_weight * (cur_index - left_start + 1) + val * (samples[cur_index] - samples[left_start]));
//      }
      best_neg = std::min(best_neg, left_sum);
      cur_sum = 0.0;
      cur_start = cur_index;
    } else {
      cur_sum += hist_val;
    }
    cur_sum -= sample_weight;
    best_neg = std::min(best_neg, cur_sum);
  }

//  if (std::abs(best) > std::abs(best_neg)) {
//    printf("pos\n");
//    return best;
//  } else {
//    printf("neg\n");
//    return -best_neg;
//  }
  return std::max(best, -best_neg);
}*/


bool histogram_merging(const std::vector<double>& samples,
                       double domain_left,
                       double domain_right,
                       int num_initial_intervals,
                       int num_merged_intervals_holdout,
                       int max_final_num_intervals,
                       std::vector<HistogramInterval>* result,
                       int* num_merging_iterations,
                       int* num_a1_computations) {

  // Step 0: initialization
  *num_merging_iterations = 0;
  *num_a1_computations = 0;
  std::vector<HistogramInterval> v1(num_initial_intervals);
  std::vector<HistogramInterval> v2(num_initial_intervals);
  std::vector<HistogramInterval> candidate_intervals(num_initial_intervals);
  std::vector<std::pair<double, int>> errors1(num_initial_intervals);
  std::vector<std::pair<double, int>> errors2(num_initial_intervals);
  int n = samples.size();
  double sample_weight = 1.0 / n;

  // Step 1: build initial intervals
  int samples_per_interval = n / num_initial_intervals;
  int num_extra_samples = n - samples_per_interval * num_initial_intervals;
  double last_boundary = domain_left;
  int last_sample_index = 0;

  for (int ii = 0; ii < num_initial_intervals; ++ii) {
    HistogramInterval& cur_interval = v1[ii];
    cur_interval.left = last_boundary;
    cur_interval.left_sample_index = last_sample_index;
    cur_interval.right_sample_index = last_sample_index + samples_per_interval;
    if (ii < num_extra_samples) {
      cur_interval.right_sample_index += 1;
    }
    if (ii == num_initial_intervals - 1) {
      cur_interval.right = domain_right;
    } else {
      cur_interval.right = (samples[cur_interval.right_sample_index - 1]
                            + samples[cur_interval.right_sample_index]) / 2.0;
    }
    cur_interval.value = compute_interval_hypothesis(cur_interval,
        sample_weight);
    last_boundary = cur_interval.right;
    last_sample_index = cur_interval.right_sample_index;
  }

  // Step 2: merging
  int cur_num_intervals = num_initial_intervals;
  int prev_num_intervals = 0;
  std::vector<HistogramInterval>* cur_intervals_p = &v1;
  std::vector<HistogramInterval>* prev_intervals_p = &v2;
  std::vector<HistogramInterval>* tmp_intervals_p;
  
  while (cur_num_intervals > max_final_num_intervals) {
    //printf("\ncur_num_intervals: %d\n", cur_num_intervals);

    // Step 2.0: update references
    *num_merging_iterations += 1;
    prev_num_intervals = cur_num_intervals;
    tmp_intervals_p = prev_intervals_p;
    prev_intervals_p = cur_intervals_p;
    cur_intervals_p = tmp_intervals_p;
    std::vector<HistogramInterval>& cur_intervals = *cur_intervals_p;
    std::vector<HistogramInterval>& prev_intervals = *prev_intervals_p;
    
    // Step 2.1: computing errors
    int num_candidates = prev_num_intervals / 2;
    for (int ii = 0; ii < num_candidates; ++ii) {
      candidate_intervals[ii].left = prev_intervals[2 * ii].left;
      candidate_intervals[ii].right = prev_intervals[2 * ii + 1].right;
      candidate_intervals[ii].left_sample_index =
          prev_intervals[2 * ii].left_sample_index;
      candidate_intervals[ii].right_sample_index =
          prev_intervals[2 * ii + 1].right_sample_index;
      candidate_intervals[ii].value = compute_interval_hypothesis(
          candidate_intervals[ii], sample_weight);
      double err = compute_a1_interval_error(candidate_intervals[ii], samples);
      /*double err2 = compute_a1_interval_error2(candidate_intervals[ii], samples);
      if (std::abs(err - err2) > 0.0000001) {
        printf("ERROR: A1 distance incorrect for ii = %d. Result %lf but should be %lf\n", ii, err, err2);
      }*/
      errors1[ii] = std::make_pair(err, ii);
      *num_a1_computations += 1;
    }

    // Step 2.2: find threshold
    std::copy(errors1.begin(),
              errors1.begin() + num_candidates,
              errors2.begin());
    int threshold_pos = num_candidates - num_merged_intervals_holdout;
    std::nth_element(errors2.begin(),
                     errors2.begin() + threshold_pos,
                     errors2.begin() + num_candidates);
    const std::pair<double, int>& error_threshold = errors2[threshold_pos];
   
    // Step 2.3: build new hypothesis 
    cur_num_intervals = 0;
    for (int ii = 0; ii < num_candidates; ++ii) {
      if (errors1[ii] >= error_threshold) {
        cur_intervals[cur_num_intervals] = prev_intervals[2 * ii];
        cur_intervals[cur_num_intervals + 1] = prev_intervals[2 * ii + 1];
        cur_num_intervals += 2;
      } else {
        cur_intervals[cur_num_intervals] = candidate_intervals[ii];
        cur_num_intervals += 1;
      }
    }
    if (prev_num_intervals % 2 == 1) {
      cur_intervals[cur_num_intervals] = prev_intervals[prev_num_intervals - 1];
      cur_num_intervals += 1;
    }
  }

  // Step 3: return result
  result->resize(cur_num_intervals);
  std::copy(cur_intervals_p->begin(),
            cur_intervals_p->begin() + cur_num_intervals,
            result->begin());

  return true;
}


}  // namespace density_estimation


#endif
