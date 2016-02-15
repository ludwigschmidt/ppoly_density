#ifndef __PRIORITY_QUEUE_H__
#define __PRIORITY_QUEUE_H__

#include <map>
#include <set>
#include <vector>

namespace density_estimation {

template <typename ValueType, typename IndexType>
class PriorityQueue {
 public:
  PriorityQueue(int max_size) : data(max_size), index_to_location(max_size),
                                num_elements(0) {}

  void get_min(ValueType* value, IndexType* index) {
    *value = data[0].first;
    *index = data[0].second;
  }

  void insert_unsorted(ValueType value, IndexType index) {
    data[num_elements].first = value;
    data[num_elements].second = index;
    index_to_location[index] = num_elements;
    num_elements += 1;
  }

  void make_heap() {
    int rightmost = parent(num_elements - 1);
    for (int cur_loc = rightmost; cur_loc >= 0; --cur_loc) {
      heap_down(cur_loc);
    }
  }

  void update_key(ValueType new_value, IndexType index) {
    int cur_loc = index_to_location[index];
    ValueType old_val = data[cur_loc].first;
    data[cur_loc].first = new_value;
    if (new_value < old_val) {
      heap_up(cur_loc);
    } else {
      heap_down(cur_loc);
    }
  }

  void delete_element(IndexType index) {
    int cur_loc = index_to_location[index];
    ValueType old_val = data[cur_loc].first;
    index_to_location[index] = -1;
    data[cur_loc] = data[num_elements - 1];
    index_to_location[data[cur_loc].second] = cur_loc;
    num_elements -= 1;
    if (data[cur_loc].first < old_val) {
      heap_up(cur_loc);
    } else {
      heap_down(cur_loc);
    }
  }
 
 private:
  int lchild(int x) {
    return 2 * x + 1;
  }

  int rchild(int x) {
    return 2 * x + 2;
  }

  int parent(int x) {
    return (x - 1) / 2;
  }

  void swap_data(int a, int b) {
    std::pair<ValueType, IndexType> tmp = data[a];
    data[a] = data[b];
    data[b] = tmp;
    index_to_location[data[a].second] = a;
    index_to_location[data[b].second] = b;
  }

  void heap_up(int cur_loc) {
    int p = parent(cur_loc);
    while (cur_loc > 0 && data[p].first > data[cur_loc].first) {
      swap_data(p, cur_loc);
      cur_loc = p;
      p = parent(cur_loc);
    }
  }

  void heap_down(int cur_loc) {
    while (true) {
      int lc = lchild(cur_loc);
      int rc = rchild(cur_loc);
      if (lc >= num_elements) {
        return;
      }

      if (data[cur_loc].first <= data[lc].first) {
        if (rc >= num_elements || data[cur_loc].first <= data[rc].first) {
          return;
        } else {
          swap_data(cur_loc, rc);
          cur_loc = rc;
        }
      } else {
        if (rc >= num_elements || data[lc].first <= data[rc].first) {
          swap_data(cur_loc, lc);
          cur_loc = lc;
        } else {
          swap_data(cur_loc, rc);
          cur_loc = rc;
        }
      }
    }
  }

  std::vector<std::pair<ValueType, IndexType>> data;
  std::vector<int> index_to_location;
  int num_elements;
};

}  // namespace density_estimation

#endif
