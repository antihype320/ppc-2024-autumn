// Copyright 2023 Nesterov Alexander
#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace tsatsyn_a_increasing_contrast_by_histogram_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_data;
  std::vector<int> res;
};

}  // namespace tsatsyn_a_increasing_contrast_by_histogram_seq