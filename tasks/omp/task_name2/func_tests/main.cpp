// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include "omp/task_name2/include/ops_omp.hpp"

TEST(task_name2_omp, Test_Sum_10) {
  std::vector<double> C(3 * 3, 0.0);
  std::vector<double> A = {2, 0, 2, 0, 0, 1, 0, 2, 0, 0, 3, 0};
  std::vector<double> B = {2, 0, 2, 0, 0, 0, 0, 1, 0, 0, 0, 3};
  std::vector<double> res = {4, 2, 4, 0, 0, 6, 0, 3, 0};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs_count.emplace_back(3);
  taskDataSeq->inputs_count.emplace_back(A.size() / 3);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs_count.emplace_back(4);
  taskDataSeq->inputs_count.emplace_back(A.size() / 4);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(C.data()));
  taskDataSeq->outputs_count.emplace_back(C.size());

  // Create Task
  TaskName2OpenMP testTaskOpenMP(taskDataSeq);
  ASSERT_EQ(testTaskOpenMP.validation(), true);
  testTaskOpenMP.pre_processing();
  testTaskOpenMP.run();
  testTaskOpenMP.post_processing();

  for (size_t i = 0; i < C.size(); i++) {
    EXPECT_DOUBLE_EQ(res[i], C[i]);
  }
}

TEST(task_name2_omp, Test_Sum_20) {
  std::vector<double> C(3 * 3, 0.0);
  std::vector<double> A = {2, 0, 2, 0, 0, 1, 0, 2, -2, 0, 3, 0};
  std::vector<double> B = {2, 0, 0, 0, -3, 0, 0, 1, 4, 1, 2, 3};
  std::vector<double> res = {4, 2, 8, 2, 1, 6, -4, 3, 12};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs_count.emplace_back(3);
  taskDataSeq->inputs_count.emplace_back(A.size() / 3);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs_count.emplace_back(4);
  taskDataSeq->inputs_count.emplace_back(A.size() / 4);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(C.data()));
  taskDataSeq->outputs_count.emplace_back(C.size());

  // Create Task
  TaskName2OpenMP testTaskOpenMP(taskDataSeq);
  ASSERT_EQ(testTaskOpenMP.validation(), true);
  testTaskOpenMP.pre_processing();
  testTaskOpenMP.run();
  testTaskOpenMP.post_processing();

  for (size_t i = 0; i < C.size(); i++) {
    EXPECT_DOUBLE_EQ(res[i], C[i]);
  }
}

TEST(task_name2_omp, Test_Sum_50) {
  int n = 15, m = 15, k = 15;
  double ro = 0.3;
  std::vector<double> A(n * m, 0.0);
  TaskName2OpenMP::genrateSparseMatrix(A.data(), A.size(), ro);
  std::vector<double> B(m * k, 0.0);
  TaskName2OpenMP::genrateSparseMatrix(B.data(), B.size(), ro);
  std::vector<double> C(n * k, 0.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->inputs_count.emplace_back(m);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs_count.emplace_back(m);
  taskDataSeq->inputs_count.emplace_back(k);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(C.data()));
  taskDataSeq->outputs_count.emplace_back(C.size());

  // Create Task
  TaskName2OpenMP testTaskOpenMP(taskDataSeq);
  ASSERT_EQ(testTaskOpenMP.validation(), true);
  testTaskOpenMP.pre_processing();
  testTaskOpenMP.run();
  testTaskOpenMP.post_processing();

  // for (size_t i = 0; i < C.size(); i++) {
  //   std::cout << C[i] << " ";
  // }
  // std::cout << std::endl;
}

TEST(task_name2_omp, Test_Sum_70) {
  int n = 50, m = 50, k = 50;
  double ro = 0.3;
  std::vector<double> A(n * m, 0.0);
  TaskName2OpenMP::genrateSparseMatrix(A.data(), A.size(), ro);
  std::vector<double> B(m * k, 0.0);
  TaskName2OpenMP::genrateSparseMatrix(B.data(), B.size(), ro);
  std::vector<double> C(n * k, 0.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->inputs_count.emplace_back(m);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs_count.emplace_back(m);
  taskDataSeq->inputs_count.emplace_back(k);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(C.data()));
  taskDataSeq->outputs_count.emplace_back(C.size());

  // Create Task
  TaskName2OpenMP testTaskOpenMP(taskDataSeq);
  ASSERT_EQ(testTaskOpenMP.validation(), true);
  testTaskOpenMP.pre_processing();
  testTaskOpenMP.run();
  testTaskOpenMP.post_processing();

  // std::sort(out.begin(), out.end());
  // std::sort(hull.begin(), hull.end());
  // for (size_t i = 0; i < hull.size(); i++) {
  //   EXPECT_DOUBLE_EQ(hull[i].x, out[i].x);
  //   EXPECT_DOUBLE_EQ(hull[i].y, out[i].y);
  // }
}

TEST(task_name2_omp, Test_Sum_100) {
  // int n = 5000, h = 600;
  // std::vector<jarvis::r> points(n);
  // std::vector<jarvis::r> hull(h);
  // std::vector<jarvis::r> out(hull.size());

  // jarvis::prepare_points(points, hull.data(), hull.size(), 45.0);

  // // Create TaskData
  // std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  // taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
  // taskDataSeq->inputs_count.emplace_back(points.size());
  // taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  // taskDataSeq->outputs_count.emplace_back(out.size());

  // // Create Task
  // TaskName2OpenMP testTaskOpenMP(taskDataSeq);
  // ASSERT_EQ(testTaskOpenMP.validation(), true);
  // testTaskOpenMP.pre_processing();
  // testTaskOpenMP.run();
  // testTaskOpenMP.post_processing();

  // std::sort(out.begin(), out.end());
  // std::sort(hull.begin(), hull.end());
  // for (size_t i = 0; i < hull.size(); i++) {
  //   EXPECT_DOUBLE_EQ(hull[i].x, out[i].x);
  //   EXPECT_DOUBLE_EQ(hull[i].y, out[i].y);
  // }
}
