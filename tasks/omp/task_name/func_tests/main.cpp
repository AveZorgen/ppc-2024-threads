// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include "omp/task_name/include/ops_omp.hpp"

TEST(task_name_omp, Test_Sum_10) {
  std::vector<jarvis::r> points = {{2, 0}, {2, 2}, {1, 1}, {0, 2}, {0, 0}};
  std::vector<jarvis::r> hull = {{2, 0}, {2, 2}, {0, 2}, {0, 0}};
  std::vector<jarvis::r> out(hull.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
  taskDataSeq->inputs_count.emplace_back(points.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  TaskNameOpenMP testTaskOpenMP(taskDataSeq);
  ASSERT_EQ(testTaskOpenMP.validation(), true);
  testTaskOpenMP.pre_processing();
  testTaskOpenMP.run();
  testTaskOpenMP.post_processing();

  std::sort(out.begin(), out.end());
  std::sort(hull.begin(), hull.end());
  for (size_t i = 0; i < hull.size(); i++) {
    EXPECT_DOUBLE_EQ(hull[i].x, out[i].x);
    EXPECT_DOUBLE_EQ(hull[i].y, out[i].y);
  }
}

TEST(task_name_omp, Test_Sum_20) {
  int n = 10, h = 5;
  std::vector<jarvis::r> points(n);
  std::vector<jarvis::r> hull(h);
  std::vector<jarvis::r> out(hull.size());

  jarvis::prepare_points(points.data(), points.size(), hull.data(), hull.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
  taskDataSeq->inputs_count.emplace_back(points.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  TaskNameOpenMP testTaskOpenMP(taskDataSeq);
  ASSERT_EQ(testTaskOpenMP.validation(), true);
  testTaskOpenMP.pre_processing();
  testTaskOpenMP.run();
  testTaskOpenMP.post_processing();

  std::sort(out.begin(), out.end());
  std::sort(hull.begin(), hull.end());
  for (size_t i = 0; i < hull.size(); i++) {
    EXPECT_DOUBLE_EQ(hull[i].x, out[i].x);
    EXPECT_DOUBLE_EQ(hull[i].y, out[i].y);
  }
}

TEST(task_name_omp, Test_Sum_50) {
  int n = 100, h = 60;
  std::vector<jarvis::r> points(n);
  std::vector<jarvis::r> hull(h);
  std::vector<jarvis::r> out(hull.size());

  jarvis::prepare_points(points.data(), points.size(), hull.data(), hull.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
  taskDataSeq->inputs_count.emplace_back(points.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  TaskNameOpenMP testTaskOpenMP(taskDataSeq);
  ASSERT_EQ(testTaskOpenMP.validation(), true);
  testTaskOpenMP.pre_processing();
  testTaskOpenMP.run();
  testTaskOpenMP.post_processing();

  std::sort(out.begin(), out.end());
  std::sort(hull.begin(), hull.end());
  for (size_t i = 0; i < hull.size(); i++) {
    EXPECT_DOUBLE_EQ(hull[i].x, out[i].x);
    EXPECT_DOUBLE_EQ(hull[i].y, out[i].y);
  }
}

TEST(task_name_omp, Test_Sum_70) {
  int n = 1000, h = 600;
  std::vector<jarvis::r> points(n);
  std::vector<jarvis::r> hull(h);
  std::vector<jarvis::r> out(hull.size());

  jarvis::prepare_points(points.data(), points.size(), hull.data(), hull.size(), 9.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
  taskDataSeq->inputs_count.emplace_back(points.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  TaskNameOpenMP testTaskOpenMP(taskDataSeq);
  ASSERT_EQ(testTaskOpenMP.validation(), true);
  testTaskOpenMP.pre_processing();
  testTaskOpenMP.run();
  testTaskOpenMP.post_processing();

  std::sort(out.begin(), out.end());
  std::sort(hull.begin(), hull.end());
  for (size_t i = 0; i < hull.size(); i++) {
    EXPECT_DOUBLE_EQ(hull[i].x, out[i].x);
    EXPECT_DOUBLE_EQ(hull[i].y, out[i].y);
  }
}

TEST(task_name_omp, Test_Sum_100) {
  int n = 5000, h = 600;
  std::vector<jarvis::r> points(n);
  std::vector<jarvis::r> hull(h);
  std::vector<jarvis::r> out(hull.size());

  jarvis::prepare_points(points.data(), points.size(), hull.data(), hull.size(), 45.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
  taskDataSeq->inputs_count.emplace_back(points.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  TaskNameOpenMP testTaskOpenMP(taskDataSeq);
  ASSERT_EQ(testTaskOpenMP.validation(), true);
  testTaskOpenMP.pre_processing();
  testTaskOpenMP.run();
  testTaskOpenMP.post_processing();

  std::sort(out.begin(), out.end());
  std::sort(hull.begin(), hull.end());
  for (size_t i = 0; i < hull.size(); i++) {
    EXPECT_DOUBLE_EQ(hull[i].x, out[i].x);
    EXPECT_DOUBLE_EQ(hull[i].y, out[i].y);
  }
}
