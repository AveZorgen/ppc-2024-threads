// Copyright 2024 Nesterov Alexander
#include "seq/task_name4/include/ops_seq.hpp"

#include <iostream>
#include <map>
#include <random>
#include <stack>
#include <utility>
#include <vector>

bool task4::TestTaskSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->inputs_count[0] > 0 && taskData->inputs_count[1] > 0;
}

bool task4::TestTaskSequential::pre_processing() {
  internal_order_test();
  width = taskData->inputs_count[0];
  height = taskData->inputs_count[1];
  auto* input = reinterpret_cast<int*>(taskData->inputs[0]);
  bin_image = std::vector<int>(input, input + width * height);
  out_hull = {};
  return true;
}

bool task4::TestTaskSequential::run() {
  internal_order_test();
  for (auto component : label_components()) {
    for (auto p : graham(component)) {
      // std::cout << p.x << "," << p.y << " ";
      out_hull.emplace_back(p.x);
      out_hull.emplace_back(p.y);
    }
    // std::cout << "\n";
    out_hull.emplace_back(SEPARATOR);
  }
  return true;
}

bool task4::TestTaskSequential::post_processing() {
  internal_order_test();
  std::copy(out_hull.begin(), out_hull.end(), reinterpret_cast<int*>(taskData->outputs[0]));
  return true;
}

std::vector<std::vector<task4::pnt>> task4::TestTaskSequential::label_components() {
  std::vector<std::vector<task4::pnt>> comps;
  const int visited = 42;
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      if (bin_image[i * width + j] == 1) {
        std::vector<task4::pnt> comp;
        comp.push_back({j, i});
        bin_image[i * width + j] = visited;

        std::stack<task4::pnt> st;
        st.emplace(j, i);

        while (!st.empty()) {
          int x = st.top().x;
          int y = st.top().y;
          st.pop();

          for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
              int nx = x + dx;
              int ny = y + dy;
              if (0 <= nx && nx < width && 0 <= ny && ny < height && bin_image[ny * width + nx] == 1) {
                comp.push_back({nx, ny});
                bin_image[ny * width + nx] = visited;
                st.push({nx, ny});
              }
            }
          }
        }
        comps.emplace_back(comp);
      }
    }
  }
  return comps;
}

inline int rotate(task4::pnt a, task4::pnt b, task4::pnt c) {
  return (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x);
}
inline double dist(task4::pnt a, task4::pnt b) { return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2)); }
inline double det(task4::pnt a, task4::pnt b, task4::pnt c) {
  return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

bool cmp(task4::pnt left_point, task4::pnt a, task4::pnt b) {
  const double det_ = det(left_point, a, b);
  if (det_ == 0) return dist(left_point, a) < dist(left_point, b);
  return det_ > 0;
}

std::vector<task4::pnt> task4::TestTaskSequential::graham(std::vector<task4::pnt> points) {
  if (points.size() < 3) return points;

  int left_index = 0;
  task4::pnt left_point = points[left_index];

  for (size_t i = 1; i < points.size(); i++) {
    if (points[i].x < left_point.x || (points[i].x == left_point.x && points[i].y < left_point.y)) {
      left_point = points[i];
      left_index = i;
    }
  }

  std::swap(points[left_index], points[points.size() - 1]);
  points.pop_back();

  std::sort(points.begin(), points.end(),
            [left_point](const task4::pnt& a, const task4::pnt& b) { return cmp(left_point, a, b); });

  std::vector<task4::pnt> hull = {left_point, points[0]};
  for (size_t i = 1; i < points.size(); i++) {
    const auto sz = hull.size();
    const auto c = points[i];

    const double rotate_ = rotate(hull[sz - 2], hull[sz - 1], c);
    if (rotate_ == 0) {
      hull[sz - 1] = c;
    } else {
      if (rotate_ < 0) {
        while (rotate(hull[hull.size() - 2], hull[hull.size() - 1], c) <= 0) {
          hull.pop_back();
        }
      }
      hull.push_back(c);
    }
  }

  return hull;
}
