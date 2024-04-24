// Copyright 2024 Mamaeva Olga
#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include "omp/mamaeva_o_jarvis/include/ops_omp.hpp"

using namespace mamaeva_omp;

TEST(mamaeva_o_jarvis_omp, test3_hull_i) {
  std::vector<r> points = {{2, 0}, {2, 2}, {1, 1}};
  std::vector<r> hull = {{2, 0}, {2, 2}, {1, 1}};
  std::vector<r> out(hull.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
  taskData->inputs_count.emplace_back(points.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskData->outputs_count.emplace_back(out.size());

  // Create Task
  JarvisHullOmp testTask(taskData);
  ASSERT_EQ(testTask.validation(), true);
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();

  std::sort(out.begin(), out.end());
  std::sort(hull.begin(), hull.end());
  for (size_t i = 0; i < hull.size(); i++) {
    ASSERT_NEAR(hull[i].x, out[i].x, 1e-3);
    ASSERT_NEAR(hull[i].y, out[i].y, 1e-3);
  }
}

TEST(mamaeva_o_jarvis_omp, test4_hull_i) {
  std::vector<r> points = {{2, 0}, {2, 2}, {1, 1}, {-1, 2}};
  std::vector<r> hull = {{2, 0}, {2, 2}, {-1, 2}};
  std::vector<r> out(hull.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
  taskData->inputs_count.emplace_back(points.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskData->outputs_count.emplace_back(out.size());

  // Create Task
  JarvisHullOmp testTask(taskData);
  ASSERT_EQ(testTask.validation(), true);
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();

  std::sort(out.begin(), out.end());
  std::sort(hull.begin(), hull.end());
  for (size_t i = 0; i < hull.size(); i++) {
    ASSERT_NEAR(hull[i].x, out[i].x, 1e-3);
    ASSERT_NEAR(hull[i].y, out[i].y, 1e-3);
  }
}

TEST(mamaeva_o_jarvis_omp, test5_hull_i) {
  std::vector<r> points = {{2, 0}, {2, 2}, {1, 1}, {0, 2}, {0, 0}};
  std::vector<r> hull = {{2, 0}, {2, 2}, {0, 2}, {0, 0}};
  std::vector<r> out(hull.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
  taskData->inputs_count.emplace_back(points.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskData->outputs_count.emplace_back(out.size());

  // Create Task
  JarvisHullOmp testTask(taskData);
  ASSERT_EQ(testTask.validation(), true);
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();

  std::sort(out.begin(), out.end());
  std::sort(hull.begin(), hull.end());
  for (size_t i = 0; i < hull.size(); i++) {
    ASSERT_NEAR(hull[i].x, out[i].x, 1e-3);
    ASSERT_NEAR(hull[i].y, out[i].y, 1e-3);
  }
}

TEST(mamaeva_o_jarvis_omp, test5_hull_d) {
  std::vector<r> points = {{0, 0}, {0.5, 0.5}, {0, 1}, {0, 0.5}, {-0.5, 0.5}};
  std::vector<r> hull = {{0, 0}, {0.5, 0.5}, {0, 1}, {-0.5, 0.5}};
  std::vector<r> out(hull.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
  taskData->inputs_count.emplace_back(points.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskData->outputs_count.emplace_back(out.size());

  // Create Task
  JarvisHullOmp testTask(taskData);
  ASSERT_EQ(testTask.validation(), true);
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();

  std::sort(out.begin(), out.end());
  std::sort(hull.begin(), hull.end());
  for (size_t i = 0; i < hull.size(); i++) {
    ASSERT_NEAR(hull[i].x, out[i].x, 1e-3);
    ASSERT_NEAR(hull[i].y, out[i].y, 1e-3);
  }
}

TEST(mamaeva_o_jarvis_omp, testrnd_hull_d) {
  int n = 100;
  int h = 60;
  std::vector<r> points(n);
  std::vector<r> hull(h);
  std::vector<r> out(hull.size());

  prepare_points(points.data(), points.size(), hull.data(), hull.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
  taskData->inputs_count.emplace_back(points.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskData->outputs_count.emplace_back(out.size());

  // Create Task
  JarvisHullOmp testTask(taskData);
  ASSERT_EQ(testTask.validation(), true);
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();

  std::sort(out.begin(), out.end());
  std::sort(hull.begin(), hull.end());
  for (size_t i = 0; i < hull.size(); i++) {
    ASSERT_NEAR(hull[i].x, out[i].x, 1e-3);
    ASSERT_NEAR(hull[i].y, out[i].y, 1e-3);
  }
}
