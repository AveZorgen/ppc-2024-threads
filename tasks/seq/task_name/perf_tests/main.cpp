// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/task_name/include/ops_seq.hpp"

TEST(sequential_task_name_perf_test, test_pipeline_run) {
  int n = 20000, h = 1200;
  std::vector<jarvis::r> points(n);
  std::vector<jarvis::r> hull(h);
  std::vector<jarvis::r> out(hull.size());

  jarvis::prepare_points(points.data(), points.size(), hull.data(), hull.size(), 250.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
  taskDataSeq->inputs_count.emplace_back(points.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  auto testTaskSequential = std::make_shared<TaskNameSequential>(taskDataSeq);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  std::sort(out.begin(), out.end());
  std::sort(hull.begin(), hull.end());
  for (size_t i = 0; i < hull.size(); i++) {
    EXPECT_DOUBLE_EQ(hull[i].x, out[i].x);
    EXPECT_DOUBLE_EQ(hull[i].y, out[i].y);
  }
}

TEST(sequential_task_name_perf_test, test_task_run) {
  int n = 20000, h = 1200;
  std::vector<jarvis::r> points(n);
  std::vector<jarvis::r> hull(h);
  std::vector<jarvis::r> out(hull.size());

  jarvis::prepare_points(points.data(), points.size(), hull.data(), hull.size(), 250.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
  taskDataSeq->inputs_count.emplace_back(points.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  auto testTaskSequential = std::make_shared<TaskNameSequential>(taskDataSeq);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  std::sort(out.begin(), out.end());
  std::sort(hull.begin(), hull.end());
  for (size_t i = 0; i < hull.size(); i++) {
    EXPECT_DOUBLE_EQ(hull[i].x, out[i].x);
    EXPECT_DOUBLE_EQ(hull[i].y, out[i].y);
  }
}
