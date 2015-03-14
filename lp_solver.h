#ifndef __LP_SOLVER_H__
#define __LP_SOLVER_H__


#include <glpk.h>

namespace density_estimation {

class LPSolver {
 public:
  LPSolver(int num_variables) : num_variables_(num_variables), num_rows_(0),
      num_rows_allocated_(0), row_indices_(num_variables + 1),
      padded_coeffs_(num_variables + 1) {
//    simplex_.setLogLevel(0);
//    simplex_.resize(0, num_variables_);
    glp_init_smcp(&params_);
    params_.msg_lev = GLP_MSG_ERR;
    lp_ = glp_create_prob();
    glp_set_obj_dir(lp_, GLP_MIN);
    glp_add_cols(lp_, num_variables);
    row_indices_[0] = 0;
    for (int ii = 1; ii <= num_variables_; ++ii) {
      glp_set_col_bnds(lp_, ii, GLP_FR, 0.0, 0.0);
      row_indices_[ii] = ii;
    }
  }

  void set_objective(const double* coeffs) {
    for (int ii = 1; ii <= num_variables_; ++ii) {
      if (coeffs[ii - 1] != 0.0) {
        glp_set_obj_coef(lp_, ii, coeffs[ii - 1]);
      }
    }
  }

  void add_constraint(const double* coeffs, double bound) {
//    printf("Added constraint: %lf %lf %lf <= %lf\n", coeffs[0], coeffs[1],
//        coeffs[2], bound);
   
    padded_coeffs_[0] = 0.0;
    for (int ii = 0; ii < num_variables_; ++ii) {
      padded_coeffs_[ii + 1] = coeffs[ii];
    }

    if (num_rows_ == num_rows_allocated_) {
      num_rows_allocated_ += 1;
      glp_add_rows(lp_, 1);
    }
    num_rows_ += 1;

    glp_set_row_bnds(lp_, num_rows_, GLP_UP, 0.0, bound);
    glp_set_mat_row(lp_, num_rows_, num_variables_, row_indices_.data(),
        padded_coeffs_.data());
  }

  void reset_constraints() {
    for (int ii = 1; ii <= num_rows_; ++ii) {
      glp_set_row_bnds(lp_, ii, GLP_FR, 0.0, 0.0);
    }
    num_rows_ = 0;
  }

  bool solve(double* coeffs) {
    int return_code = glp_simplex(lp_, &params_);
    if (return_code != 0) {
      printf("ERROR: GLPK returned %d\n", return_code);
      return false;
    }
    for (int ii = 1; ii <= num_variables_; ++ii) {
      coeffs[ii - 1] = glp_get_col_prim(lp_, ii);
    }
    return true;
  }
 
 private:
  int num_variables_;
  int num_rows_;
  int num_rows_allocated_;
  glp_prob* lp_;
  glp_smcp params_;
  std::vector<int> row_indices_;
  std::vector<double> padded_coeffs_;
};

}  // namespace density_estimation


/*
#include "ClpSimplex.hpp"

namespace density_estimation {

class LPSolver {
 public:
  LPSolver(int num_variables) : num_variables_(num_variables),
      row_indices_(num_variables) {
    simplex_.setLogLevel(0);
    simplex_.resize(0, num_variables_);
    for (int ii = 0; ii < num_variables_; ++ii) {
      simplex_.setColumnBounds(ii, -COIN_DBL_MAX, COIN_DBL_MAX);
      row_indices_[ii] = ii;
    }
  }

  void set_objective(const double* coeffs) {
    for (int ii = 0; ii < num_variables_; ++ii) {
      if (coeffs[ii] != 0.0) {
        simplex_.setObjectiveCoefficient(ii, coeffs[ii]);
      }
    }
  }

  void add_constraint(const double* coeffs, double bound) {
    //printf("Added constraint: %lf %lf %lf <= %lf\n", coeffs[0], coeffs[1],
    //    coeffs[2], bound);
    simplex_.addRow(num_variables_, row_indices_.data(), coeffs, -COIN_DBL_MAX,
        bound);
  }

  void reset_constraints() {
    simplex_.resize(0, num_variables_);
//    for (int ii = 0; ii < num_variables_; ++ii) {
//      simplex_.setColumnBounds(ii, COIN_DBL_MIN, COIN_DBL_MAX);
//    }
  }

  bool solve(double* coeffs) {
    try {
      simplex_.primal();
      double* tmp = simplex_.primalColumnSolution();
      for (int ii = 0; ii < num_variables_; ++ii) {
        coeffs[ii] = tmp[ii];
      }
      return true;
    } catch (CoinError e) {
      e.print();
      return false;
    }
  }
 
 private:
  int num_variables_;
  ClpSimplex  simplex_;
  std::vector<int> row_indices_;
};

}  // namespace density_estimation
*/

#endif
