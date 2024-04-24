// Copyright 2024 Mamaeva Olga
#include "omp/mamaeva_o_jarvis/include/ops_omp.hpp"

#define M_PI 3.14159265358979323846
#include <omp.h>

#include <algorithm>
#include <iostream>
#include <iterator>
#include <random>
#include <utility>
#include <vector>

using namespace mamaeva_omp;

void prepare_points(r* points, int n, r* hull, int h, double r, unsigned int seed) {
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

bool JarvisHullOmp::pre_processing() {
  internal_order_test();
  auto* p = reinterpret_cast<r*>(taskData->inputs[0]);
  points = std::vector<r>(p, p + taskData->inputs_count[0]);
  hull = reinterpret_cast<r*>(taskData->outputs[0]);
  hull_sz = taskData->outputs_count[0];
  return true;
}

bool JarvisHullOmp::validation() {
  internal_order_test();
  // Check count elements of output
  return (taskData->inputs[0] != nullptr) && (taskData->inputs_count[0] != 0u) && (taskData->outputs[0] != nullptr) &&
         (taskData->outputs_count[0] != 0u);
}

void Jarvis(const std::vector<r>& points, r* hull) {
  int n = points.size();
  int left = 0;
  r p_point = points[left];
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

  int p = left;
  int q;
  int h_i = 0;
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
}

bool JarvisHullOmp::run() {
  internal_order_test();
  Jarvis(points, hull);
  return true;
}

bool JarvisHullOmp::post_processing() {
  internal_order_test();
  return true;
}
