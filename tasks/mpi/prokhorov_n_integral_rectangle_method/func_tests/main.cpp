#define _USE_MATH_DEFINES
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <cmath>
#include <vector>

#include "mpi/prokhorov_n_integral_rectangle_method/include/ops_mpi.hpp"

TEST(prokhorov_n_integral_rectangle_method_mpi, Test_Cosine) {
  boost::mpi::environment env;
  boost::mpi::communicator world;
  std::vector<double> global_input(3);
  std::vector<double> global_result(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_input[0] = 0.0;
    global_input[1] = M_PI / 2;
    global_input[2] = 1000;
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_input.data()));
    taskDataPar->inputs_count.emplace_back(global_input.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  prokhorov_n_integral_rectangle_method_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  testMpiTaskParallel.set_function([](double x) { return std::cos(x); });
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_NEAR(global_result[0], 1.0, 1e-5);
  }
}

// Добавьте аналогичные исправления для других тестов