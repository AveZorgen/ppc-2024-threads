// Copyright 2023 Nesterov Alexander
#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace task4 {
struct pnt {
  int x{};
  int y{};
  pnt() : x(0), y(0) {}
};

const int SEPARATOR = -1;

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> bin_image{};
  std::vector<int> out_hull{};
  int width{};
  int height{};

  std::vector<std::vector<pnt>> label_components();
  std::vector<pnt> graham(std::vector<pnt> points);
};
}  // namespace task4
