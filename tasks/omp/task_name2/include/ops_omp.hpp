// Copyright 2023 Nesterov Alexander
#pragma once

#include <memory>  // may be del
#include <string>
#include <utility>  // may be del
#include <vector>

#include "core/task/include/task.hpp"

const double EPS = 1e-6;

class MatrixCRS {
 public:
  int N;
  int NZ;
  std::vector<double> Value;
  std::vector<int> Col;
  std::vector<int> RowIndex;

  explicit MatrixCRS(int n = 0, int nz = 0);
  MatrixCRS(const double* matrix, int n, int m, bool transpose = false);
};

class TaskName2OpenMP : public ppc::core::Task {
 public:
  explicit TaskName2OpenMP(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
  static void genrateSparseMatrix(double* matrix, int sz, double ro);

 private:
  MatrixCRS A, BT, C;
  double* c_out{};
  int M{};
};
