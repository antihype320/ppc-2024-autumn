// Copyright 2023 Nesterov Alexander
#include "mpi/prokhorov_n_integral_rectangle_method/include/ops_mpi.hpp"

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <functional>
#include <numeric>
#include <random>
#include <vector>

namespace prokhorov_n_integral_rectangle_method_mpi {

double TestMPITaskSequential::integrate(const std::function<double(double)>& f, double left_, double right_, int n) {
  double step = (right_ - left_) / n;

  std::vector<double> areas(n);

  for (int i = 0; i < n; ++i) {
    double x = left_ + (i + 0.5) * step;
    areas[i] = f(x) * step;
  }

  return std::accumulate(areas.begin(), areas.end(), 0.0);
}

void TestMPITaskSequential::set_function(const std::function<double(double)>& func) { func_ = func; }

bool TestMPITaskSequential::pre_processing() {
  internal_order_test();
  uint8_t* inputs_raw = taskData->inputs[0];
  std::vector<double> inputs(reinterpret_cast<double*>(inputs_raw), reinterpret_cast<double*>(inputs_raw) + 3);
  left_ = inputs[0];
  right_ = inputs[1];
  n = static_cast<int>(inputs[2]);
  res = 0.0;
  return true;
}

bool TestMPITaskSequential::validation() {
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

bool TestMPITaskSequential::run() {
  internal_order_test();
  res = integrate(func_, left_, right_, n);
  return true;
}

bool TestMPITaskSequential::post_processing() {
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
double TestMPITaskParallel::parallel_integrate(const std::function<double(double)>& f, double left_, double right_,
                                               int n, const boost::mpi::communicator& world) {
  double step = (right_ - left_) / n;
  double local_area = 0.0;
  int local_n = n / world.size();
  int start = world.rank() * local_n;
  int end = (world.rank() + 1) * local_n;

  if (end > n) {
    end = n;
  }

  for (int i = start; i < end; ++i) {
    double x = left_ + (i + 0.5) * step;
    local_area += f(x) * step;
  }

  return local_area;
}

void TestMPITaskParallel::set_function(const std::function<double(double)>& func) { func_ = func; }

bool TestMPITaskParallel::pre_processing() {
  internal_order_test();
  auto* inputs = reinterpret_cast<double*>(taskData->inputs[0]);
  left_ = inputs[0];
  right_ = inputs[1];
  n = static_cast<int>(inputs[2]);

  boost::mpi::broadcast(world, left_, 0);
  boost::mpi::broadcast(world, right_, 0);
  boost::mpi::broadcast(world, n, 0);

  global_res = 0.0;
  return true;
}

bool TestMPITaskParallel::validation() {
  internal_order_test();

  if (taskData->inputs_count[0] != 3) {
    if (world.rank() == 0) {
      std::cerr << "Error: Incorrect number of inputs. Expected 3, got " << taskData->inputs_count[0] << std::endl;
    }
    return false;
  }

  if (taskData->outputs_count[0] != 1) {
    if (world.rank() == 0) {
      std::cerr << "Error: Incorrect number of outputs. Expected 1, got " << taskData->outputs_count[0] << std::endl;
    }
    return false;
  }

  auto inputs = reinterpret_cast<double*>(taskData->inputs[0]);
  if (inputs[0] >= inputs[1]) {
    if (world.rank() == 0) {
      std::cerr << "Error: Left boundary must be less than right boundary." << std::endl;
    }
    return false;
  }

  if (static_cast<int>(inputs[2]) <= 0) {
    if (world.rank() == 0) {
      std::cerr << "Error: Number of intervals must be greater than 0." << std::endl;
    }
    return false;
  }

  return true;
}

bool TestMPITaskParallel::run() {
  internal_order_test();

  if (n <= 0) {
    std::cerr << "Invalid number of intervals." << std::endl;
    return false;
  }

  local_res = parallel_integrate(func_, left_, right_, n, world);

  boost::mpi::all_reduce(world, local_res, global_res, std::plus<double>());

  return true;
}

bool TestMPITaskParallel::post_processing() {
  internal_order_test();

  if (std::isnan(global_res) || std::isinf(global_res)) {
    std::cerr << "Error: Integration result is not a valid number. Cannot proceed with post-processing." << std::endl;
    return false;
  }

  if (world.rank() == 0 && taskData->outputs_count[0] != 1) {
    std::cerr << "Error: Incorrect number of outputs. Expected 1, got " << taskData->outputs_count[0] << std::endl;
    return false;
  }

  if (world.rank() == 0) {
    try {
      reinterpret_cast<double*>(taskData->outputs[0])[0] = global_res;
    } catch (const std::exception& e) {
      std::cerr << "Error during post-processing: " << e.what() << std::endl;
      return false;
    }
  }

  return true;
}

}  // namespace prokhorov_n_integral_rectangle_method_mpi
