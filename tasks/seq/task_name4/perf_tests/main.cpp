// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/task_name4/include/ops_seq.hpp"

TEST(sequential_task_name4_perf_test, test_pipeline_run) {
  int width = 2000;
  int height = 2000;

  // Create data
  std::vector<int> out(width * height);
  std::vector<int> in(width * height, 0);
  int sqw = width / 2;
  int sqh = height / 2;
  for (int i = 0; i < sqh - 1; i++) {
    for (int j = 0; j < sqw - 1; j++) {
      in[i * width + j] = 1;
      in[i * width + (sqw + 1 + j)] = 1;
    }
  }
  for (int i = 0; i < width; i++) {
    in[sqh * width + i] = 1;
    in[(sqh + 2) * width + i] = 1;
  }
  for (int i = sqh + 3; i < height; i++) {
    in[i * width + sqw - 1] = 1;
    in[i * width + sqw] = 1;
  }

  std::vector<int> hullTrue = {
      0,       0,          sqw - 2,   0,          sqw - 2,          sqh - 2, 0,       sqh - 2,   task4::SEPARATOR,
      sqw + 1, 0,          width - 1, 0,          width - 1,        sqh - 2, sqw + 1, sqh - 2,   task4::SEPARATOR,
      0,       sqh,        width - 1, sqh,        task4::SEPARATOR, 0,       sqh + 2, width - 1, sqh + 2,
      sqw,     height - 1, sqw - 1,   height - 1, task4::SEPARATOR};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  // Create Task
  auto testTaskSequential = std::make_shared<task4::TestTaskSequential>(taskDataSeq);

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

  for (size_t i = 0; i < hullTrue.size(); i++) ASSERT_EQ(hullTrue[i], out[i]);
}

TEST(sequential_task_name4_perf_test, test_task_run) {
  int width = 2000;
  int height = 2000;

  // Create data
  std::vector<int> out(width * height);
  std::vector<int> in(width * height, 0);
  int sqw = width / 2;
  int sqh = height / 2;
  for (int i = 0; i < sqh - 1; i++) {
    for (int j = 0; j < sqw - 1; j++) {
      in[i * width + j] = 1;
      in[i * width + (sqw + 1 + j)] = 1;
    }
  }
  for (int i = 0; i < width; i++) {
    in[sqh * width + i] = 1;
    in[(sqh + 2) * width + i] = 1;
  }
  for (int i = sqh + 3; i < height; i++) {
    in[i * width + sqw - 1] = 1;
    in[i * width + sqw] = 1;
  }

  std::vector<int> hullTrue = {
      0,       0,          sqw - 2,   0,          sqw - 2,          sqh - 2, 0,       sqh - 2,   task4::SEPARATOR,
      sqw + 1, 0,          width - 1, 0,          width - 1,        sqh - 2, sqw + 1, sqh - 2,   task4::SEPARATOR,
      0,       sqh,        width - 1, sqh,        task4::SEPARATOR, 0,       sqh + 2, width - 1, sqh + 2,
      sqw,     height - 1, sqw - 1,   height - 1, task4::SEPARATOR};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  // Create Task
  auto testTaskSequential = std::make_shared<task4::TestTaskSequential>(taskDataSeq);

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

  for (size_t i = 0; i < hullTrue.size(); i++) ASSERT_EQ(hullTrue[i], out[i]);
}
