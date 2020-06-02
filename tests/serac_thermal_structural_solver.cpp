// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gtest/gtest.h>

#include <fstream>

#include "coefficients/linear_coefficient.hpp"
#include "mfem.hpp"
#include "serac_config.hpp"
#include "solvers/thermal_structural_solver.hpp"

void   InitialDeformation(const mfem::Vector &x, mfem::Vector &y);
void   InitialVelocity(const mfem::Vector &x, mfem::Vector &v);
double InitialTemperature(const mfem::Vector &x);

TEST(dynamic_solver, dyn_solve)
{
  MPI_Barrier(MPI_COMM_WORLD);

  // mesh
  std::string base_mesh_file = std::string(SERAC_SRC_DIR) + "/data/beam-hex.mesh";
  const char *mesh_file      = base_mesh_file.c_str();

  // Open the mesh
  std::ifstream imesh(mesh_file);
  auto          mesh = std::make_shared<mfem::Mesh>(imesh, 1, 1, true);
  imesh.close();

  mesh->UniformRefinement();

  auto pmesh = std::make_shared<mfem::ParMesh>(MPI_COMM_WORLD, *mesh);

  int dim = pmesh->Dimension();

  // define a boundary attribute array and initialize to 0
  std::vector<int> ess_bdr(pmesh->bdr_attributes.Max(), 0);

  // boundary attribute 1 (index 0) is fixed (Dirichlet)
  ess_bdr[0] = 1;

  auto deform = std::make_shared<mfem::VectorFunctionCoefficient>(dim, InitialDeformation);
  auto velo   = std::make_shared<mfem::VectorFunctionCoefficient>(dim, InitialVelocity);
  auto temp   = std::make_shared<mfem::FunctionCoefficient>(InitialTemperature);
  auto kappa = std::make_shared<mfem::ConstantCoefficient>(0.5);

  // initialize the dynamic solver object
  ThermalStructuralSolver ts_solver(1, pmesh);
  ts_solver.SetDisplacementBCs(ess_bdr, deform);
  ts_solver.SetHyperelasticMaterialParameters(0.25, 5.0);
  ts_solver.SetConductivity(kappa);
  ts_solver.SetDisplacement(*deform);
  ts_solver.SetVelocity(*velo);
  ts_solver.SetTemperature(*temp);
  ts_solver.SetTimestepper(TimestepMethod::SDIRK33);
  ts_solver.SetCouplingScheme(CouplingScheme::OperatorSplit);

  // Make a temperature-dependent viscosity
  //auto temp_gf_coef = std::make_shared<mfem::GridFunctionCoefficient>(ts_solver.GetTemperature()->gf.get());
  //LinearTransformationCoefficient visc_coef(temp_gf_coef, 0.001, 1.0);
  auto visc = std::make_shared<mfem::ConstantCoefficient>(0.0);
  ts_solver.SetViscosity(visc);


  // Set the linear solver parameters
  LinearSolverParameters params;
  params.prec        = Preconditioner::BoomerAMG;
  params.abs_tol     = 1.0e-8;
  params.rel_tol     = 1.0e-4;
  params.max_iter    = 500;
  params.lin_solver  = LinearSolver::GMRES;
  params.print_level = 0;

  // Set the nonlinear solver parameters
  NonlinearSolverParameters nl_params;
  nl_params.rel_tol     = 1.0e-4;
  nl_params.abs_tol     = 1.0e-8;
  nl_params.print_level = 1;
  nl_params.max_iter    = 500;
  ts_solver.SetSolidSolverParameters(params, nl_params);
  ts_solver.SetThermalSolverParameters(params);

  // Initialize the VisIt output
  ts_solver.InitializeOutput(OutputType::VisIt, "dynamic_thermal_solid");

  // Construct the internal dynamic solver data structures
  ts_solver.CompleteSetup();

  double t       = 0.0;
  double t_final = 6.0;
  double dt      = 1.0;

  // Ouput the initial state
  ts_solver.OutputState();

  // Perform time-integration
  // (looping over the time iterations, ti, with a time-step dt).
  bool last_step = false;
  for (int ti = 1; !last_step; ti++) {
    double dt_real = std::min(dt, t_final - t);
    t += dt_real;
    last_step = (t >= t_final - 1e-8 * dt);

    ts_solver.AdvanceTimestep(dt_real);
  }

  // Output the final state
  ts_solver.OutputState();

  // Check the final displacement and velocity L2 norms
  mfem::Vector zero(dim);
  zero = 0.0;
  mfem::VectorConstantCoefficient zerovec(zero);

  double v_norm    = ts_solver.GetVelocity()->gf->ComputeLpError(2.0, zerovec);
  double x_norm    = ts_solver.GetDisplacement()->gf->ComputeLpError(2.0, zerovec);
  double temp_norm = ts_solver.GetTemperature()->gf->ComputeLpError(2.0, zerovec);

  EXPECT_NEAR(12.86733, x_norm, 0.0001);
  EXPECT_NEAR(0.22298, v_norm, 0.0001);

  MPI_Barrier(MPI_COMM_WORLD);
}

int main(int argc, char *argv[])
{
  int result = 0;

  ::testing::InitGoogleTest(&argc, argv);

  MPI_Init(&argc, &argv);

  result = RUN_ALL_TESTS();

  MPI_Finalize();

  return result;
}

void InitialDeformation(const mfem::Vector &x, mfem::Vector &y)
{
  y    = x;
  y(1) = x(0) * 0.01;
}

void InitialVelocity(__attribute__((unused)) const mfem::Vector &x, mfem::Vector &v) { v = 0.0; }

double InitialTemperature(const mfem::Vector &x)
{
  double temp = 2.0;
  if (x(0) < 1.0) {
    temp = 5.0;
  }
  return temp;
}