// Copyright 2024 Nesterov Alexander
#include "omp/task_name/include/ops_omp.hpp"

#define _USE_MATH_DEFINES
#include <math.h>
#include <omp.h>

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

bool TaskNameOpenMP::pre_processing() {
  internal_order_test();
  jarvis::r* p = reinterpret_cast<jarvis::r*>(taskData->inputs[0]);
  points = std::vector<jarvis::r>(p, p + taskData->inputs_count[0]);
  hull = reinterpret_cast<jarvis::r*>(taskData->outputs[0]);
  hull_sz = taskData->outputs_count[0];
  return true;
}

bool TaskNameOpenMP::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->inputs[0] && taskData->inputs_count[0] && taskData->outputs[0] && taskData->outputs_count[0];
}

int Jarvis(const std::vector<jarvis::r>& points, jarvis::r* hull) {
  int n = points.size();
  int left = 0;
  jarvis::r p_point = points[left];
  int i;

#pragma omp parallel shared(left) private(i) firstprivate(n)
  {
    int l_local = 0;
#pragma omp for nowait
    for (i = 1; i < n; i++) {
      if (points[i] < points[l_local]) {
        l_local = i;
      }
    }

#pragma omp critical
    {
      if (points[l_local] < points[left]) left = l_local;
    }
  }

  int p = left, q, h_i = 0;
  while (true) {
    q = (p + 1) % n;
    p_point = points[p];

#pragma omp parallel shared(q) private(i) firstprivate(n)
    {
      int q_local = q;
#pragma omp for nowait
      for (i = 0; i < n; i++) {
        if (((p_point - points[i]) ^ (points[q_local] - points[i])) < -1e-7) q_local = i;
      }

#pragma omp critical
      {
        if (((p_point - points[q_local]) ^ (points[q] - points[q_local])) < -1e-7) q = q_local;
      }
    }

    hull[h_i++] = points[q];
    p = q;
    if (p == left) break;
  }
  return h_i - 1;
}

bool TaskNameOpenMP::run() {
  internal_order_test();
  int h_i = Jarvis(points, hull);
  return h_i == hull_sz;
}

bool TaskNameOpenMP::post_processing() {
  internal_order_test();
  return true;
}
