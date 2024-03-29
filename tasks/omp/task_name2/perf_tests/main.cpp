// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>
#include <omp.h>

#include <algorithm>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "omp/task_name2/include/ops_omp.hpp"

TEST(openmp_task_name2_perf_test, test_pipeline_run) {
  int n = 2000, m = 2000, k = 2000; double ro = 0.005;
  std::vector<double> A(n*m, 0.0); TaskName2OpenMP::genrateSparseMatrix(A.data(), A.size(), ro);
  std::vector<double> B(m*k, 0.0); TaskName2OpenMP::genrateSparseMatrix(B.data(), B.size(), ro);
  std::vector<double> C(n*k, 0.0);

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
  auto testTaskOpenMP = std::make_shared<TaskName2OpenMP>(taskDataSeq);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  perfAttr->current_timer = [&] { return omp_get_wtime(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskOpenMP);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
}

TEST(openmp_task_name2_perf_test, test_task_run) {
  int n = 2000, m = 2000, k = 2000; double ro = 0.005;
  std::vector<double> A(n*m, 0.0); TaskName2OpenMP::genrateSparseMatrix(A.data(), A.size(), ro);
  std::vector<double> B(m*k, 0.0); TaskName2OpenMP::genrateSparseMatrix(B.data(), B.size(), ro);
  std::vector<double> C(n*k, 0.0);

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
  auto testTaskOpenMP = std::make_shared<TaskName2OpenMP>(taskDataSeq);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  perfAttr->current_timer = [&] { return omp_get_wtime(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskOpenMP);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
}
