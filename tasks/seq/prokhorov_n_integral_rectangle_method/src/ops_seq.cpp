// Copyright 2024 Nesterov Alexander
#include "seq/prokhorov_n_integral_rectangle_method/include/ops_seq.hpp"

#include <algorithm>
#include <functional>
#include <numeric>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

bool prokhorov_n_integral_rectangle_method::TestTaskSequential::pre_processing() {
  internal_order_test();
  uint8_t* inputs_raw = taskData->inputs[0];

  std::vector<double> inputs(reinterpret_cast<double*>(inputs_raw), reinterpret_cast<double*>(inputs_raw) + 3);

  left_ = inputs[0];
  right_ = inputs[1];
  n = static_cast<int>(inputs[2]);

  res = 0.0;
  return true;
}

bool prokhorov_n_integral_rectangle_method::TestTaskSequential::validation() {
  internal_order_test();

  if (taskData->inputs_count[0] != 3) {
    std::cerr << "Error: Incorrect number of inputs. Expected 3, got " << taskData->inputs_count[0] << std::endl;
    return false;
  }

  if (taskData->outputs_count[0] != 1) {
    std::cerr << "Error: Incorrect number of outputs. Expected 1, got " << taskData->outputs_count[0] << std::endl;
    return false;
  }

  auto inputs = reinterpret_cast<double*>(taskData->inputs[0]);
  if (inputs[0] >= inputs[1]) {
    std::cerr << "Error: Left boundary must be less than right boundary." << std::endl;
    return false;
  }

  if (static_cast<int>(inputs[2]) <= 0) {
    std::cerr << "Error: Number of intervals must be greater than 0." << std::endl;
    return false;
  }

  return true;
}

bool prokhorov_n_integral_rectangle_method::TestTaskSequential::run() {
  internal_order_test();
  res = integrate(func_, left_, right_, n);
  return true;
}

bool prokhorov_n_integral_rectangle_method::TestTaskSequential::post_processing() {
  internal_order_test();

  if (std::isnan(res) || std::isinf(res)) {
    std::cerr << "Error: Integration result is not a valid number. Cannot proceed with post-processing." << std::endl;
    return false;
  }

  if (taskData->outputs_count[0] != 1) {
    std::cerr << "Error: Incorrect number of outputs. Expected 1, got " << taskData->outputs_count[0] << std::endl;
    return false;
  }

  try {
    reinterpret_cast<double*>(taskData->outputs[0])[0] = res;
  } catch (const std::exception& e) {
    std::cerr << "Error during post-processing: " << e.what() << std::endl;
    return false;
  }

  return true;
}

double prokhorov_n_integral_rectangle_method::TestTaskSequential::integrate(const std::function<double(double)>& f,
                                                                            double left_, double right_, int n) {
  double step = (right_ - left_) / n;
  std::vector<double> areas(n);

  for (int i = 0; i < n; ++i) {
    double x = left_ + (i + 0.5) * step;
    areas[i] = f(x) * step;
  }

  return std::accumulate(areas.begin(), areas.end(), 0.0);
}

void prokhorov_n_integral_rectangle_method::TestTaskSequential::set_function(
    const std::function<double(double)>& func) {
  func_ = func;
}

void prokhorov_n_integral_rectangle_method::TestTaskSequential::set_boundaries(double left, double right) {
  left_ = left;
  right_ = right;
}

void prokhorov_n_integral_rectangle_method::TestTaskSequential::set_num_steps(int steps) {
  if (steps > 0) {
    n = steps;
  } else {
    std::cerr << "Number of steps must be positive!" << std::endl;
  }
}
