// Copyright 2024 Nesterov Alexander
#include "seq/task_name/include/ops_seq.hpp"

#define M_PI 3.14159265358979323846

#include <algorithm>
#include <iostream>
#include <iterator>
#include <random>
#include <utility>
#include <vector>

void jarvis::prepare_points(jarvis::r* points, int n, jarvis::r* hull, int h, double r, unsigned int seed) {
  std::mt19937 g(seed);
  std::uniform_real_distribution gen(-r, r);
  int i = 0;
  double phi = 2 * M_PI / h;
  for (i = 0; i < n; i++) {
    points[i] = {gen(g), gen(g)};
  }
  int k = n / (h + 1);
  for (i = 0; i < h; i++) {
    hull[i] = points[k * (i + 1)] = {cos(phi * i) * 2 * r, sin(phi * i) * 2 * r};
  }
}

bool TaskNameSequential::pre_processing() {
  internal_order_test();
  jarvis::r* p = reinterpret_cast<jarvis::r*>(taskData->inputs[0]);
  points = std::vector<jarvis::r>(p, p + taskData->inputs_count[0]);
  hull = reinterpret_cast<jarvis::r*>(taskData->outputs[0]);
  hull_sz = taskData->outputs_count[0];
  return true;
}

bool TaskNameSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->inputs[0] && taskData->inputs_count[0] && taskData->outputs[0] && taskData->outputs_count[0];
}

int Jarvis(const std::vector<jarvis::r>& points, jarvis::r* hull) {
  int n = points.size();
  int left = 0;
  jarvis::r p_point = points[left];
  for (int i = 1; i < n; i++)
    if (points[i] < p_point) {
      p_point = points[left = i];
    }

  int p = left, q, h_i = 0;
  while (true) {
    q = (p + 1) % n;
    for (int i = 0; i < n; i++) {
      if (((p_point - points[i]) ^ (points[q] - points[i])) < -1e-7) q = i;
    }
    hull[h_i++] = points[q];
    p_point = points[p = q];
    if (p == left) break;
  }
  return h_i - 1;
}

bool TaskNameSequential::run() {
  internal_order_test();
  // std::cout << "points:\n";
  // for (int i = 0; i < points.size(); i++) {
  //   std::cout << points[i].x << " " << points[i].y << std::endl;
  // }
  // std::cout << "hull_inp:\n";
  // for (int i = 0; i < hull_sz; i++) {
  //   std::cout << hull[i].x << " " << hull[i].y << std::endl;
  // }
  int h_i = Jarvis(points, hull);
  // std::cout << "out:\n";
  // for (int i = 0; i < hull_sz; i++) {
  //   std::cout << hull[i].x << " " << hull[i].y << std::endl;
  // }
  return h_i == hull_sz;
}

bool TaskNameSequential::post_processing() {
  internal_order_test();
  return true;
}
