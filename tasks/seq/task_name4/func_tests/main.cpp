// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "seq/task_name4/include/ops_seq.hpp"

TEST(task_name4_seq, Test1) {
  int width = 3;
  int height = 3;

  // Create data
  std::vector<int> out(width * height);
  std::vector<int> in = {0, 1, 0, 1, 0, 1, 0, 0, 0};

  std::vector<int> hullTrue = {0, 1, 1, 0, 2, 1, task4::SEPARATOR};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));

  // Create Task
  task4::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  for (size_t i = 0; i < hullTrue.size(); i++) EXPECT_EQ(hullTrue[i], out[i]);
}

TEST(task_name4_seq, Test2) {
  int width = 3;
  int height = 3;

  // Create data
  std::vector<int> out(width * height);
  std::vector<int> in = {0, 1, 0, 1, 1, 1, 0, 1, 0};

  std::vector<int> hullTrue = {0, 1, 1, 0, 2, 1, 1, 2, task4::SEPARATOR};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));

  // Create Task
  task4::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  for (size_t i = 0; i < hullTrue.size(); i++) EXPECT_EQ(hullTrue[i], out[i]);
}

TEST(task_name4_seq, Test3) {
  int width = 6;
  int height = 6;

  // Create data
  std::vector<int> out(width * height);
  std::vector<int> in = {0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0,
                         0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0};

  std::vector<int> hullTrue = {0, 2, 1, 1, 2, 2, 1, 3, task4::SEPARATOR, 3, 4, 4, 2, 5, 4, task4::SEPARATOR};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));

  // Create Task
  task4::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  for (size_t i = 0; i < hullTrue.size(); i++) EXPECT_EQ(hullTrue[i], out[i]);
}


TEST(task_name4_seq, Test4) {
  int width = 5;
  int height = 8;

  // Create data
  std::vector<int> out(width * height);
  std::vector<int> in = {0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0};

  std::vector<int> hullTrue = {2, 2, 3, 1, 4, 2, task4::SEPARATOR, 0, 6, 1, 6, 1, 7, 0, 7, task4::SEPARATOR};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));

  // Create Task
  task4::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  for (size_t i = 0; i < hullTrue.size(); i++) ASSERT_EQ(hullTrue[i], out[i]);
}

TEST(task_name4_seq, Test5) {
  int width = 10;
  int height = 10;

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
  task4::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  for (size_t i = 0; i < hullTrue.size(); i++) ASSERT_EQ(hullTrue[i], out[i]);
}
