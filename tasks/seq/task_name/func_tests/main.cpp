// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include "seq/task_name/include/ops_seq.hpp"

TEST(task_name_seq, Test_Sum_10) {
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
  TaskNameSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  std::sort(out.begin(), out.end());
  std::sort(hull.begin(), hull.end());
  for (size_t i = 0; i < hull.size(); i++) {
    ASSERT_NEAR(hull[i].x, out[i].x, 1e-3);
    ASSERT_NEAR(hull[i].y, out[i].y, 1e-3);
  }
}

TEST(task_name_seq, Test_Sum_20) {
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
  TaskNameSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  std::sort(out.begin(), out.end());
  std::sort(hull.begin(), hull.end());
  for (size_t i = 0; i < hull.size(); i++) {
    ASSERT_NEAR(hull[i].x, out[i].x, 1e-3);
    ASSERT_NEAR(hull[i].y, out[i].y, 1e-3);
  }
}

TEST(task_name_seq, Test_Sum_50) {
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
  TaskNameSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  std::sort(out.begin(), out.end());
  std::sort(hull.begin(), hull.end());
  for (size_t i = 0; i < hull.size(); i++) {
    ASSERT_NEAR(hull[i].x, out[i].x, 1e-3);
    ASSERT_NEAR(hull[i].y, out[i].y, 1e-3);
  }
}

TEST(task_name_seq, Test_Sum_70) {
  int n = 500, h = 300;
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
  TaskNameSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  std::sort(out.begin(), out.end());
  std::sort(hull.begin(), hull.end());
  for (size_t i = 0; i < hull.size(); i++) {
    ASSERT_NEAR(hull[i].x, out[i].x, 1e-3);
    ASSERT_NEAR(hull[i].y, out[i].y, 1e-3);
  }
}

TEST(task_name_seq, Test_Sum_100) {
  int n = 500, h = 600;
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
  TaskNameSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  std::sort(out.begin(), out.end());
  std::sort(hull.begin(), hull.end());
  for (size_t i = 0; i < hull.size(); i++) {
    ASSERT_NEAR(hull[i].x, out[i].x, 1e-3);
    ASSERT_NEAR(hull[i].y, out[i].y, 1e-3);
  }
}
