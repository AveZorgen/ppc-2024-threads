// Copyright 2024 Nesterov Alexander
#include "seq/task_name2/include/ops_seq.hpp"

#define _USE_MATH_DEFINES
#include <math.h>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <random>
#include <sstream>
#include <utility>
#include <vector>

MatrixCRS::MatrixCRS(int n, int nz) : N(n), NZ(nz) {
  Value.reserve(nz);
  Col.reserve(nz);
  RowIndex.reserve(n + 1);
}

MatrixCRS::MatrixCRS(const double* matrix, int n, int m, bool transpose) {
  int index;
  if (transpose) {
    N = m;
    m = n;
  } else {
    N = n;
  }
  RowIndex.reserve(N + 1);
  // std::cout << N + 1 << " " << RowIndex.size() << " ";
  NZ = 0;
  for (int i = 0; i < N; i++) {
    RowIndex[i] = NZ;
    for (int j = 0; j < m; j++) {
      index = transpose ? i + N * j : i * m + j;
      if (abs(matrix[index]) >= EPS) {
        Value.push_back(matrix[index]);
        Col.push_back(j);
        NZ++;
      }
    }
  }
  RowIndex[N] = NZ;
  // std::cout << RowIndex.size() << std::endl;
}

MatrixCRS Multiplicate(const MatrixCRS& A, const MatrixCRS& BT) {
  int N = A.N;  //! A.n == BT.N
  std::vector<int> columns;
  std::vector<double> values;
  std::vector<int> row_index;
  int rowNZ;
  int nz_expected = A.NZ * BT.NZ / N + 1;
  std::cout << "NZ = " << nz_expected << std::endl;
  values.reserve(nz_expected);
  columns.reserve(nz_expected);
  row_index.push_back(0);
  for (int i = 0; i < N; i++) {
    rowNZ = 0;
    for (int j = 0; j < N; j++) {
      double sum = 0;

      // int ks = A.RowIndex[i];
      // int ls = BT.RowIndex[j];
      // int kf = A.RowIndex[i + 1] - 1;
      // int lf = BT.RowIndex[j + 1] - 1;
      // while (ks <= kf && ls <= lf) {
      //   if (A.Col[ks] < BT.Col[ls])
      //     ks++;
      //   else if (A.Col[ks] > BT.Col[ls])
      //     ls++;
      //   else {
      //     sum += A.Value[ks] * BT.Value[ls];
      //     ks++;
      //     ls++;
      //   }
      // }

      for (int k = A.RowIndex[i]; k < A.RowIndex[i + 1]; k++)
        for (int l = BT.RowIndex[j]; l < BT.RowIndex[j + 1]; l++)
          if (A.Col[k] == BT.Col[l]) {
            sum += A.Value[k] * BT.Value[l];
            break;
          }

      if (fabs(sum) > EPS) {
        columns.push_back(j);
        values.push_back(sum);
        rowNZ++;
      }
    }
    row_index.push_back(rowNZ + row_index[i]);
  }
  MatrixCRS C(N, columns.size());
  for (int j = 0; j < columns.size(); j++) {
    C.Col[j] = columns[j];
    C.Value[j] = values[j];
  }
  for (int i = 0; i <= N; i++) C.RowIndex[i] = row_index[i];
  return C;
}

MatrixCRS Multiplicate2(const MatrixCRS& A, const MatrixCRS& BT, int m) {
  int N = A.N;  //! A.n == BT.N
  int nz_expected = A.NZ * BT.NZ / N + 1;
  std::vector<int> columns;
  std::vector<double> values;
  std::vector<int> row_index(N + 1);
  // std::cout << "NZ = " << nz_expected << std::endl;
  values.reserve(nz_expected);
  columns.reserve(nz_expected);
  std::vector<int> temp(m);
  int nz = 0;

  for (int i = 0; i < N; i++) {
    row_index[i] = nz;

    memset(temp.data(), 0, m * sizeof(*temp.data()));
    int ind1 = A.RowIndex[i], ind2 = A.RowIndex[i + 1];
    for (int k = ind1; k < ind2; k++) {
      temp[A.Col[k]] = k + 1;
    }

    for (int j = 0; j < N; j++) {
      double sum = 0;

      int ind3 = BT.RowIndex[j], ind4 = BT.RowIndex[j + 1];
      for (int k = ind3; k < ind4; k++) {
        int aind = temp[BT.Col[k]];
        if (aind != 0) {
          sum += A.Value[aind - 1] * BT.Value[k];
        }
      }

      if (fabs(sum) > EPS) {
        columns.push_back(j);
        values.push_back(sum);
        nz++;
      }
    }
  }
  row_index[N] = nz;

  MatrixCRS C(N, nz);
  std::move(values.begin(), values.end(), C.Value.begin());
  std::move(columns.begin(), columns.end(), C.Col.begin());
  std::move(row_index.begin(), row_index.end(), C.RowIndex.begin());
  return C;
}

void PrintCRSMatrix(const MatrixCRS& matrix, int m) {
  for (int i = 0; i < matrix.N; i++) {
    int j = 0;
    for (int c_j = matrix.RowIndex[i]; c_j < matrix.RowIndex[i + 1]; c_j++) {
      while (j < matrix.Col[c_j]) {
        std::cout << "0 ";
        j++;
      }
      std::cout << matrix.Value[c_j] << " ";
      j++;
    }
    while (j < m) {
      std::cout << "0 ";
      j++;
    }
    std::cout << std::endl;
  }
}

bool TaskName2Sequential::pre_processing() {
  internal_order_test();
  M = taskData->inputs_count[1];
  A = MatrixCRS(reinterpret_cast<double*>(taskData->inputs[0]), taskData->inputs_count[0], M, false);
  BT = MatrixCRS(reinterpret_cast<double*>(taskData->inputs[1]), taskData->inputs_count[2], taskData->inputs_count[3],
                 true);
  c_out = reinterpret_cast<double*>(taskData->outputs[0]);

  // transpose BT
  return true;
}

bool TaskName2Sequential::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->inputs[0] && taskData->inputs_count[0] && taskData->outputs[0] && taskData->outputs_count[0];
}

bool TaskName2Sequential::run() {
  internal_order_test();
  // std::cout << "A.Col: ";
  // for (auto a : A.Col) std::cout << a << " ";
  // std::cout << std::endl;
  // std::cout << "A.Value: ";
  // for (auto a : A.Value) std::cout << a << " ";
  // std::cout << std::endl;
  // std::cout << "A.RowIndex: ";
  // for (int i = 0; i < A.N + 1; i++) std::cout << A.RowIndex[i] << " ";
  // std::cout << std::endl;
  // std::cout << std::endl;

  // std::cout << "BT.Col: ";
  // for (auto a : BT.Col) std::cout << a << " ";
  // std::cout << std::endl;
  // std::cout << "BT.Value: ";
  // for (auto a : BT.Value) std::cout << a << " ";
  // std::cout << std::endl;
  // std::cout << "BT.RowIndex: ";
  // for (int i = 0; i < BT.N + 1; i++) std::cout << BT.RowIndex[i] << " ";
  // std::cout << std::endl;
  // std::cout << std::endl;
  // PrintCRSMatrix(A, M);
  // std::cout << std::endl;
  // PrintCRSMatrix(BT, M);
  // std::cout << std::endl;
  C = Multiplicate2(A, BT, M);
  // try {
  // C = ParMultiplicate2(A, BT, M);
  // } catch (std::exception& e) {
  //   std::cout << e.what() << std::endl;
  // }
  // std::cout << "real nz: " << C.NZ << std::endl;

  // for (int i = 0; i < C.NZ; i++) std::cout << C.Col[i] << " ";
  // std::cout << std::endl;
  // for (int i = 0; i < C.NZ; i++) std::cout << C.Value[i] << " ";
  // std::cout << std::endl;
  // for (int i = 0; i < C.N + 1; i++) std::cout << C.RowIndex[i] << " ";
  // std::cout << std::endl;
  // std::cout << std::endl;
  // PrintCRSMatrix(C, A.N);
  return true;
}

bool TaskName2Sequential::post_processing() {
  internal_order_test();

  for (int i = 0; i < C.N; i++) {
    int j = 0;
    for (int c_j = C.RowIndex[i]; c_j < C.RowIndex[i + 1]; c_j++) {
      while (j < C.Col[c_j]) {
        c_out[i * A.N + j] = 0;
        j++;
      }
      c_out[i * A.N + j] = C.Value[c_j];
      j++;
    }
    while (j < A.N) {
      c_out[i * A.N + j] = 0;
      j++;
    }
  }

  return true;
}

void TaskName2Sequential::genrateSparseMatrix(double* matrix, int sz, double ro) {
  int nz = sz * ro;
  // std::cout << "gen nz = " << nz << std::endl;
  std::uniform_int_distribution<int> distribution(0, sz - 1);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> rand_r(-10.0, 10.0);
  for (int i = 0; i < nz; i++) {
    matrix[distribution(gen)] = rand_r(gen);
  }
}
