// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
// # Author: Jonathan Wong @ LLNL.

#include <gtest/gtest.h>

#include <memory>

#include "../src/coefficients/stdfunction_coefficient.hpp"
#include "../src/integrators/wrapper_integrator.hpp"
#include "../src/numerics/expr_template_ops.hpp"
#include "mfem.hpp"

using namespace std;

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  ::testing::InitGoogleTest(&argc, argv);
  int return_code = RUN_ALL_TESTS();
  MPI_Finalize();
  return return_code;
}

class NewmarkBetaTest : public ::testing::Test {
protected:
  void SetUp()
  {
    // Set up mesh
    dim = 2;
    nex = 3;
    ney = 1;

    len   = 8.;
    width = 1.;

    mfem::Mesh mesh(nex, ney, mfem::Element::QUADRILATERAL, 1, len, width);
    pmesh = std::make_shared<mfem::ParMesh>(MPI_COMM_WORLD, mesh);
    pfes  = std::make_shared<mfem::ParFiniteElementSpace>(
        pmesh.get(), new mfem::H1_FECollection(1, dim, mfem::BasisType::GaussLobatto), 1, mfem::Ordering::byNODES);
    pfes_v = std::make_shared<mfem::ParFiniteElementSpace>(
        pmesh.get(), new mfem::H1_FECollection(1, dim, mfem::BasisType::GaussLobatto), dim, mfem::Ordering::byNODES);

    pfes_l2 = std::make_shared<mfem::ParFiniteElementSpace>(pmesh.get(), new mfem::L2_FECollection(0, dim), 1,
                                                            mfem::Ordering::byNODES);
  }

  void TearDown() {}

  double                                       width, len;
  int                                          nex, ney, nez;
  int                                          dim;
  std::shared_ptr<mfem::ParMesh>               pmesh;
  std::shared_ptr<mfem::ParFiniteElementSpace> pfes;
  std::shared_ptr<mfem::ParFiniteElementSpace> pfes_v;
  std::shared_ptr<mfem::ParFiniteElementSpace> pfes_l2;
};

/**
   @brief A class to contain second order nonlinear residual information
 */
class SecondOrderResidual {
public:
  /**
     @brief A class to contain second order nonlinear residual information

     @param [in] pfes Reference to a shared memory pointer
     @param [in] density Density of the material
   */
  SecondOrderResidual(std::shared_ptr<mfem::ParFiniteElementSpace> pfes, mfem::Coefficient& density)
      : density_(density), pfes_(pfes), up_(new mfem::ParGridFunction(pfes.get()))
  {
    *up_ = 0.;

    // Create mass integrator term for dynamic problem
    auto inertial_integrator = std::make_shared<mfem::VectorMassIntegrator>(density_);
    auto nonlinear_inertial_integrator =
        std::make_shared<serac::BilinearToNonlinearFormIntegrator>(inertial_integrator);

    residual_int_.push_back(nonlinear_inertial_integrator);
  }

  /**
     @brief Take in mfem::BilinearFormIntegrator, convert it to a nonlienarform, and save it into a list of integrators

     @param [in] blfi For the time being we assume this bilinearform integrator is dependent on the displacement (u)
   */
  void addBilinearDomainIntegrator(std::unique_ptr<mfem::BilinearFormIntegrator> blfi)
  {
    auto shared_version = std::shared_ptr<mfem::BilinearFormIntegrator>(blfi.release());
    residual_int_u_.push_back(std::make_shared<serac::BilinearToNonlinearFormIntegrator>(shared_version));
  }

  /**
     @brief Take in a mfem::LinearFormIntegrator, convert it to a nonlinearform integrator, and save it to interal
     residual integrator list for terms not dependent on u or v

     @param [in] lfi LinearFormIntegrator to convert into a nonlinearform and.
   */
  void addLinearDomainIntegrator(std::unique_ptr<mfem::LinearFormIntegrator> lfi)
  {
    auto shared_version = std::shared_ptr<mfem::LinearFormIntegrator>(lfi.release());
    residual_int_.push_back(std::make_shared<serac::LinearToNonlinearFormIntegrator>(shared_version, pfes_));
  }

  /**
     @brief Take in a mfem::NonlinearFormIntegratorand save it to interal residual integrator list for terms not
     dependent on u or v

     @param [in] nlfi LinearFormIntegrator to convert into a nonlinearform and.
  */

  void addNonlinearDomainIntegrator(std::unique_ptr<mfem::NonlinearFormIntegrator> nlfi)
  {
    auto shared_version = std::shared_ptr<mfem::NonlinearFormIntegrator>(nlfi.release());
    residual_int_.push_back(shared_version);
  }

  /**
     @brief Set boundary conditions for this operator

     @param [in] local_tdofs local dofs corresponding to essential boundary conditions to set
     @param [in] vals corresponding values of essential boundary conditions
   */
  void setBoundaryConditions(mfem::Array<int> local_tdofs, mfem::Vector vals)
  {
    MFEM_VERIFY(local_tdofs.Size() == vals.Size(), "Essential true dof size != val.Size()");

    *up_ = 0.;
    up_->SetSubVector(local_tdofs, vals);

    local_ess_tdofs_ = local_tdofs;
  }

  /**
     @brief Solve nonlinear residual using newton's method

     @param [in] R A prepared ParNonlienarForm that has been assembled and is ready to be solved
     @param [in] sol_next The solution of the nonlinear residual. Currently it's assumped to be on one pargridfunction.
   */
  void solveResidual(mfem::ParNonlinearForm& R, mfem::ParGridFunction& sol_next) const
  {
    mfem::NewtonSolver newton_solver(pfes_->GetComm());
    mfem::GMRESSolver  solver(pfes_->GetComm());
    newton_solver.SetSolver(solver);
    newton_solver.SetOperator(R);

    auto Sol_next = std::unique_ptr<mfem::HypreParVector>(sol_next.GetTrueDofs());

    mfem::Vector zero;
    newton_solver.Mult(zero, *Sol_next);

    int num_iterations_taken = newton_solver.GetNumIterations();
    std::cout << "initial iterations:" << num_iterations_taken << std::endl;
    // Copy solution back
    sol_next = *Sol_next;
  }

  /// A method to get the local boundary condition indices
  const mfem::Array<int>& getLocalTDofs() { return local_ess_tdofs_; }

  /// A method to get the finite element space corresponding to the solutions
  std::shared_ptr<mfem::ParFiniteElementSpace> getParFESpace() { return pfes_; }

  /// A method to get the boundary conditions corresponding to essential boundary conditions
  mfem::ParGridFunction& getEssentialBCValues() { return *up_; }

  /// A method to get the list of nonlinearform integrators dependent on the displacement (u)
  std::vector<std::shared_ptr<mfem::NonlinearFormIntegrator>> const& getResidualIntegratorsU()
  {
    return residual_int_u_;
  }

  /// A method to get the list of nonlinearform integrators independent of u, v
  std::vector<std::shared_ptr<mfem::NonlinearFormIntegrator>> const& getResidualIntegrators() { return residual_int_; }

private:
  std::vector<std::shared_ptr<mfem::NonlinearFormIntegrator>> residual_int_u_;  //< dependent on u
  std::vector<std::shared_ptr<mfem::NonlinearFormIntegrator>> residual_int_;    //< not dependent on u or v

  std::unique_ptr<mfem::ParGridFunction> up_;  //< holds essential boundary conditions

  mfem::Array<int>                             local_ess_tdofs_;  //< local true essential boundary conditions
  std::shared_ptr<mfem::ParFiniteElementSpace> pfes_;             //< finite element space
  mfem::Coefficient&                           density_;          //< density
};

/**
   @brief A SecondOrderTimeDependentOperator that implements the newmark beta time-stepping scheme
 */
class NewmarkBetaSecondOrder : public mfem::SecondOrderTimeDependentOperator {
public:
  /**
     @brief  A SecondOrderTimeDependentOperator that implements the newmark beta time-stepping scheme

     @param [in] residual A SecondOrderResidual
     @param [in] beta The beta parameter
     @param [in] gamma The gamma parameter
   */
  NewmarkBetaSecondOrder(std::shared_ptr<SecondOrderResidual> residual, double beta, double gamma)
      : beta_(beta), gamma_(gamma), pfes_(residual->getParFESpace()), residual_(residual)
  {
  }

  /**
     @brief This "explicit" method is used by the NewmarkSolver to get the initial acceleration at timestep = 0.

     The mfem::NewmarkSolver assumes that x, dxdt, and y have the same size

     @param [in] x The "displacement" solution at t
     @param [in] dxdt The "velocity" solution at t
     @param [out] y The "acceleration" solution at t
  */
  virtual void Mult(const mfem::Vector& x, const mfem::Vector& dxdt, mfem::Vector& y) const override
  {
    // convert x, dxdt, and y into ParGridFunctions
    mfem::ParGridFunction u0(pfes_.get(), x.GetData());
    mfem::ParGridFunction v0(pfes_.get(), dxdt.GetData());

    // The size of y has not been set yet
    y.SetSize(u0.Size());
    mfem::ParGridFunction a0(pfes_.get(), y.GetData());

    // Since u0 is constant the gradient is 0.
    auto zero_grad = [](const mfem::FiniteElement&, mfem::ElementTransformation&, const mfem::DenseMatrix& elmat) {
      auto m = std::make_shared<mfem::DenseMatrix>(elmat);
      *m     = 0.;
      return m;
    };

    // Here we substitute u for u0. If we used v0 in this problem we would do the same for those integrators.
    auto substitute_u0 = [&](const mfem::FiniteElement&, mfem::ElementTransformation& Tr, const mfem::Vector& vect) {
      auto                         ret_vect = std::make_shared<mfem::Vector>(vect.Size());
      int                          e        = Tr.ElementNo;
      mfem::ParFiniteElementSpace* pfes     = u0.ParFESpace();
      mfem::Array<int>             vdofs;
      pfes->GetElementVDofs(e, vdofs);
      u0.GetSubVector(vdofs, *ret_vect);
      return ret_vect;
    };

    // Create and assemble the nonlinearform. We want to solve R(a) = 0
    mfem::ParNonlinearForm R(pfes_.get());

    // Loop over and add nonlinearintegrators independent of u.
    for (auto integ : residual_->getResidualIntegrators())
      R.AddDomainIntegrator(new serac::PointerNonlinearFormIntegrator(integ));

    // Loop over and add nonlinearintegrators dependent on u. Perform change of variables from u to a.
    for (auto integ : residual_->getResidualIntegratorsU()) {
      auto sub_non_integ =
          std::make_shared<serac::SubstitutionNonlinearFormIntegrator>(integ, substitute_u0, zero_grad);
      R.AddDomainIntegrator(new serac::PointerNonlinearFormIntegrator(sub_non_integ));
    }

    a0 = 0.;

    // Solve the residual for a(t)
    residual_->solveResidual(R, a0);
  }

  /**
     @brief This implicit method is used by the NewmarkSolver to perform time integration

     @param [in] dt1 This term is equiavelent to dt * gamma
     @param [in] x The "displacement" predictor at (t+dt)
     @param [in] dxdt The "velocity" predictor at (t+dt)
     @param [out] k The "acceleration" solution at the next time step (t+dt)
   */
  virtual void ImplicitSolve(const double, const double dt1, const mfem::Vector& x, const mfem::Vector& dxdt,
                             mfem::Vector& k) override
  {
    // x is already x_pred
    // dxdt is already dxdt_pred
    double dt = dt1 / gamma_;  // back out dt from dt1

    mfem::ParGridFunction u_pred(pfes_.get());
    mfem::ParGridFunction v_pred(pfes_.get());
    mfem::ParGridFunction a_next(pfes_.get());
    u_pred.SetFromTrueDofs(x);
    v_pred.SetFromTrueDofs(dxdt);
    a_next.SetFromTrueDofs(k);
    a_next = 0.;

    mfem::ParGridFunction u_next(pfes_.get());

    // The NewmarkBeta scheme is:
    // u_next(a_next) = u_pred + beta * dt * dt * a_next
    // v_next(a_next) = v_pred + gamma * dt * a_next
    // a_next(u_next) = (u_next - u_pred)/(beta * dt * dt)

    // Create functions to perform change of variables in terms of u, a(u) for the residual evaluation and for the
    // gradient of the residual term

    // da_next(u_next)/du_next = 1/ (beta * dt * dt)
    double dadu = 1. / (beta_ * dt * dt);

    auto substitute_a_next = [&](const mfem::FiniteElement&, mfem::ElementTransformation& Tr,
                                 const mfem::Vector& u_next) {
      auto                         a_next_vect = std::make_shared<mfem::Vector>(u_next.Size());
      int                          e           = Tr.ElementNo;
      mfem::ParFiniteElementSpace* pfes        = u_pred.ParFESpace();
      mfem::Array<int>             vdofs;
      pfes->GetElementVDofs(e, vdofs);
      mfem::Vector u_pred_vect(u_next.Size());
      u_pred.GetSubVector(vdofs, u_pred_vect);
      *a_next_vect = dadu * (u_next - u_pred_vect);
      return a_next_vect;
    };

    auto substitute_a_next_grad = [&](const mfem::FiniteElement&, mfem::ElementTransformation&,
                                      const mfem::DenseMatrix& elmat) {
      auto m = std::make_shared<mfem::DenseMatrix>(elmat);
      *m *= dadu;
      return m;
    };

    // Create and assemble the nonlinearform. We want to solve R(u) = 0
    mfem::ParNonlinearForm R(pfes_.get());

    // Loop over and add nonlinearintegrators. Perform change of variables from a to u.
    for (auto integ : residual_->getResidualIntegrators()) {
      auto sub_non_integ = std::make_shared<serac::SubstitutionNonlinearFormIntegrator>(integ, substitute_a_next,
                                                                                        substitute_a_next_grad);

      R.AddDomainIntegrator(new serac::PointerNonlinearFormIntegrator(sub_non_integ));
    }

    // Loop over and add nonlinearintegrators dependent on u.
    for (auto integ : residual_->getResidualIntegratorsU()) {
      R.AddDomainIntegrator(new serac::PointerNonlinearFormIntegrator(integ));
    }

    // Set boundary condition indices
    R.SetEssentialTrueDofs(residual_->getLocalTDofs());

    // Set boundary condition values
    u_next = residual_->getEssentialBCValues();

    // Solve R(u) = 0 to get u(t+dt)
    residual_->solveResidual(R, u_next);

    // Update a(t+dt) from u(t+dt)
    evaluate(dadu * (u_next - u_pred), a_next);
    a_next.GetTrueDofs(k);
  }

private:
  std::shared_ptr<mfem::ParFiniteElementSpace> pfes_;          // reference to finite element space to solution
  std::shared_ptr<SecondOrderResidual>         residual_;      // holds the residual
  double                                       beta_, gamma_;  //< Newmark parameters
};

TEST_F(NewmarkBetaTest, Simple)
{
  double beta  = 0.25;
  double gamma = 0.5;

  mfem::ConstantCoefficient rho(1.0);

  auto                   residual = std::make_shared<SecondOrderResidual>(pfes_v, rho);
  NewmarkBetaSecondOrder second_order(residual, beta, gamma);
  mfem::NewmarkSolver    ns(beta, gamma);
  ns.Init(second_order);

  mfem::Array<int> offsets(4);
  offsets[0] = 0;
  offsets[1] = offsets[0] + pfes_v->GetVSize();
  offsets[2] = offsets[1] + pfes_v->GetVSize();
  offsets[3] = offsets[2] + pfes_v->GetVSize();

  // u(0) = no displacements
  mfem::BlockVector     x0(offsets);
  mfem::ParGridFunction u0(pfes_v.get());
  u0 = 0.;

  // v(0) = 1 m/s in the y direction
  mfem::ParGridFunction v0(pfes_v.get());
  v0 = 0.;
  for (int i = 0; i < pfes_v->GetNV(); i++) v0[pfes_v->DofToVDof(i, 1)] = 1.;

  x0.GetBlock(0) = u0;
  x0.GetBlock(1) = v0;

  // Domain is not accelerating at t=0
  x0.GetBlock(2) = 0.;

  double dt = 0.01;
  double t  = 0.;

  std::cout << "t = " << t << std::endl;
  std::cout << "x" << std::endl;
  x0.GetBlock(0).Print();
  std::cout << "v" << std::endl;
  x0.GetBlock(1).Print();
  std::cout << "a" << std::endl;
  x0.GetBlock(2).Print();

  // Store results
  mfem::Vector u_prev(x0.GetBlock(0));
  mfem::Vector v_prev(x0.GetBlock(1));
  mfem::Vector a_prev(x0.GetBlock(1));

  mfem::Vector u_next(u_prev);
  mfem::Vector v_next(v_prev);
  second_order.Mult(u_prev, v_prev, a_prev);

  ns.Step(u_next, v_next, t, dt);

  std::cout << "t = " << t << std::endl;
  std::cout << "x" << std::endl;
  u_next.Print();
  std::cout << "v" << std::endl;
  v_next.Print();

  // back out a_next
  mfem::Vector a_next(v_next);
  a_next -= v_prev;
  a_next /= (dt * gamma);

  // Check udot
  for (int d = 0; d < u_next.Size(); d++)
    EXPECT_NEAR(u_next[d],
                u_prev[d] + dt * v_prev[d] + 0.5 * dt * dt * ((1. - 2. * beta) * a_prev[d] + 2. * beta * a_next[d]),
                std::max(1.e-4 * u_next[d], 1.e-8));

  // Check vdot
  for (int d = 0; d < v_next.Size(); d++)
    EXPECT_NEAR(v_next[d], v_prev[d] + dt * (1 - gamma) * a_prev[d] + gamma * dt * a_next[d],
                std::max(1.e-4 * v_next[d], 1.e-8));
}

void CalculateLambdaMu(double E, double nu, double& lambda, double& mu)
{
  lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
  mu     = 0.5 * E / (1.0 + nu);
}

TEST_F(NewmarkBetaTest, Equilibrium)
{
  // Problem parameters
  double E  = 17.e6;
  double nu = 0.3;
  double r  = 0.163;  // density
  double g  = -32.3;
  double lambda_c, mu_c;
  CalculateLambdaMu(E, nu, lambda_c, mu_c);

  double i              = 1. / 12. * width * width * width;
  double m              = width * r;
  double omega          = (1.875) * (1.875) * sqrt(E * i / (m * len * len * len * len));
  double period         = 2. * M_PI / omega;
  double end_time       = 2. * period;
  int    num_time_steps = 40;
  double dt             = end_time / num_time_steps;
  std::cout << "dt: " << dt << std::endl;  // 0.001939

  // tried beta = 0.5, gamma = 0.1
  double beta  = 0.25;
  double gamma = 0.5;  // 0.5

  mfem::ConstantCoefficient rho(r);
  auto                      residual = std::make_shared<SecondOrderResidual>(pfes_v, rho);
  NewmarkBetaSecondOrder    second_order(residual, beta, gamma);
  mfem::NewmarkSolver       ns(beta, gamma);
  ns.Init(second_order);

  mfem::Array<int> offsets(3);
  offsets[0] = 0;
  offsets[1] = offsets[0] + pfes_v->GetVSize();
  offsets[2] = offsets[1] + pfes_v->GetVSize();

  // u(0) = no displacements
  mfem::BlockVector     x0(offsets);
  mfem::ParGridFunction u0(pfes_v.get());
  u0 = 0.;

  mfem::ParGridFunction v0(pfes_v.get());
  v0 = 0.;

  x0.GetBlock(0) = u0;
  x0.GetBlock(1) = v0;

  // Fix x = 0
  int                           ne = nex;
  serac::StdFunctionCoefficient fixed([ne](mfem::Vector& x) { return (x[0] < 1. / ne) ? 1. : 0.; });

  mfem::Array<int> bdr_attr_list = serac::makeBdrAttributeList(*pmesh, fixed);
  for (int be = 0; be < pmesh->GetNBE(); be++) {
    pmesh->GetBdrElement(be)->SetAttribute(bdr_attr_list[be]);
  }
  pmesh->SetAttributes();

  mfem::Array<int> ess_tdof_list;
  mfem::Array<int> bdr_attr_is_ess(pmesh->bdr_attributes.Max());
  bdr_attr_is_ess = 0;
  if (bdr_attr_is_ess.Size() > 1) bdr_attr_is_ess[1] = 1;
  pfes_v->GetEssentialTrueDofs(bdr_attr_is_ess, ess_tdof_list);

  // Homogeneous dirchlet bc
  mfem::Vector u_ess(ess_tdof_list.Size());
  u_ess = 0.;

  residual->setBoundaryConditions(ess_tdof_list, u_ess);

  // Add self-weight
  mfem::Vector grav(dim);
  grav    = 0.;
  grav[1] = g;
  mfem::VectorConstantCoefficient grav_v_coef(grav);
  // Here rho is the actually rho = rho_const * vf
  mfem::ScalarVectorProductCoefficient gravity_load_coef(rho, grav_v_coef);
  auto                                 gravity2 = std::make_unique<mfem::VectorDomainLFIntegrator>(gravity_load_coef);
  residual->addLinearDomainIntegrator(std::move(gravity2));

  // Make it non rigid body
  mfem::ConstantCoefficient lambda(lambda_c);
  mfem::ConstantCoefficient mu(mu_c);
  auto                      elasticity2 = std::make_unique<mfem::ElasticityIntegrator>(lambda, mu);
  residual->addBilinearDomainIntegrator(std::move(elasticity2));

  cout << "output volume fractions :" << endl;

  mfem::ParGridFunction gravity_eval(pfes_v.get());
  gravity_eval.ProjectCoefficient(gravity_load_coef);
  gravity_eval.Print();

  double t = 0.;

  std::cout << "t = " << t << std::endl;

  // Store results
  mfem::Vector u_next(x0.GetBlock(0));
  mfem::Vector v_next(x0.GetBlock(1));

  mfem::ParGridFunction u_visit(pfes_v.get());
  mfem::ParGridFunction v_visit(pfes_v.get());
  u_visit = u_next;
  v_visit = v_next;

  mfem::VisItDataCollection visit("NewmarkBeta", pmesh.get());
  visit.RegisterField("u_next", &u_visit);
  visit.RegisterField("v_next", &v_visit);
  visit.SetCycle(0);
  visit.SetTime(0.);
  visit.Save();

  for (int i = 0; i < num_time_steps; i++) {
    ns.Step(x0.GetBlock(0), x0.GetBlock(1), t, dt);
    u_visit = x0.GetBlock(0);
    v_visit = x0.GetBlock(1);
    visit.SetTime(t);
    visit.SetCycle(i + 1);
    visit.Save();
  }
}

/**
   @brief A mfem::TimeDependentOperator that can take a SecondOrderResidual and recast it as a first order system
 */
class FirstOrderSystem : public mfem::TimeDependentOperator {
public:
  /**
   @brief A mfem::TimeDependentOperator that can take a SecondOrderResidual and recast it as a first order system

   @param [in] residual The second order residual to solve and integrate

  */
  FirstOrderSystem(std::shared_ptr<SecondOrderResidual> residual)
      : mfem::TimeDependentOperator(residual->getParFESpace()->GetTrueVSize() * 2, 0.,
                                    mfem::TimeDependentOperator::Type::IMPLICIT),
        offsets_(3),
        pfes_(residual->getParFESpace()),
        residual_(residual)
  {
    offsets_[0] = 0;
    offsets_[1] = offsets_[0] + pfes_->GetTrueVSize();
    offsets_[2] = offsets_[1] + pfes_->GetTrueVSize();
  }

  /// The explicit implementation of this time dependent operator
  virtual void Mult(const mfem::Vector&, mfem::Vector&) const
  {
    mfem::mfem_error("not currently supported for this second order wrapper");
  };

  /**
     @brief The implicit time stepping solve for the recast Second Order Residual as a first order system

     @param[in] dt The current time step.
     @param[in] x The solution at time t
     @param[out] k The rate at time (t+dt)
   */
  virtual void ImplicitSolve(const double dt, const mfem::Vector& x, mfem::Vector& k) override
  {
    MFEM_VERIFY(x.Size() == offsets_[2], "vectors are not the same size");
    mfem::BlockVector bx(x.GetData(), offsets_);

    /* A second order p.d.e can be recast as a first order system
      u_next = u_prev + dt * v_next
      v_next = v_prev + dt * a_next

      Thie means:
      u_next = u_prev + dt * (v_prev + dt * a_next);
      u_next = u_prev + dt * v_prev + dt*dt*a_next
      a_next(u_next) = (u_next - u_prev - dt * v_prev)/(dt*dt)
    */

    mfem::ParGridFunction u_prev(pfes_.get());
    mfem::ParGridFunction v_prev(pfes_.get());
    u_prev.SetFromTrueDofs(bx.GetBlock(0));
    v_prev.SetFromTrueDofs(bx.GetBlock(1));

    mfem::ParGridFunction a_next(pfes_.get());
    mfem::ParGridFunction u_next(pfes_.get());
    a_next = 0.;

    // Create a function to perform a change of variables in terms of u,  a(u)
    auto substitute_a = [&](const mfem::FiniteElement&, mfem::ElementTransformation& Tr,
                            const mfem::Vector& u_next_elem) {
      auto                         a_next_elem = std::make_shared<mfem::Vector>(u_next_elem.Size());
      int                          e           = Tr.ElementNo;
      mfem::ParFiniteElementSpace* pfes        = u_prev.ParFESpace();
      mfem::Array<int>             vdofs;
      pfes->GetElementVDofs(e, vdofs);

      mfem::Vector u_prev_elem(u_prev.Size());
      mfem::Vector v_prev_elem(u_prev.Size());

      u_prev.GetSubVector(vdofs, u_prev_elem);
      v_prev.GetSubVector(vdofs, v_prev_elem);

      *a_next_elem = (u_next_elem - u_prev_elem - v_prev_elem * dt) / (dt * dt);

      return a_next_elem;
    };

    // Create a function to perform a change of variables a(u) for the gradient of the residual
    double dadu              = 1. / (dt * dt);
    auto   substitute_a_grad = [&](const mfem::FiniteElement&, mfem::ElementTransformation&,
                                 const mfem::DenseMatrix& elmat) {
      auto m = std::make_shared<mfem::DenseMatrix>(elmat);
      *m *= dadu;
      return m;
    };

    // Create and assemble the nonlinearform. We want to solve R(u) = 0
    mfem::ParNonlinearForm R(pfes_.get());

    // Loop over and add nonlinearintegrators. Perform change of variables to u.
    for (auto integ : residual_->getResidualIntegrators()) {
      auto sub_non_integ =
          std::make_shared<serac::SubstitutionNonlinearFormIntegrator>(integ, substitute_a, substitute_a_grad);

      R.AddDomainIntegrator(new serac::PointerNonlinearFormIntegrator(sub_non_integ));
    }

    // Loop over and add nonlinearintegrators dependent on u.
    for (auto integ : residual_->getResidualIntegratorsU()) {
      R.AddDomainIntegrator(new serac::PointerNonlinearFormIntegrator(integ));
    }

    // Set boundary condition indices
    R.SetEssentialTrueDofs(residual_->getLocalTDofs());

    // Set boundary condition values
    u_next = residual_->getEssentialBCValues();

    // Solve R(u) = 0 to get u(t+dt)
    residual_->solveResidual(R, u_next);

    // Update a(t+dt) from u(t+dt)
    evaluate(1. / (dt * dt) * (u_next - u_prev - dt * v_prev), a_next);

    // v_next = v_old + dt * a_next
    mfem::ParGridFunction v_next(pfes_.get());
    v_next = v_prev + dt * a_next;

    // set the results back
    mfem::BlockVector bk(k.GetData(), offsets_);
    v_next.GetTrueDofs(bk.GetBlock(0));
    a_next.GetTrueDofs(bk.GetBlock(1));
  }

private:
  std::shared_ptr<mfem::ParFiniteElementSpace> pfes_;
  std::shared_ptr<SecondOrderResidual>         residual_;
  mfem::Array<int>                             offsets_;
};

TEST_F(NewmarkBetaTest, Equilibrium_firstorder)
{
  // Problem parameters
  double E  = 17.e6;
  double nu = 0.3;
  double r  = 0.163;  // density
  double g  = -32.3;
  double lambda_c, mu_c;
  CalculateLambdaMu(E, nu, lambda_c, mu_c);

  double i              = 1. / 12. * width * width * width;
  double m              = width * r;
  double omega          = (1.875) * (1.875) * sqrt(E * i / (m * len * len * len * len));
  double period         = 2. * M_PI / omega;
  double end_time       = 2. * period;
  int    num_time_steps = 40;
  double dt             = end_time / num_time_steps;

  double beta  = 0.5;
  double gamma = 1.;

  mfem::ConstantCoefficient rho(r);
  auto                      residual = std::make_shared<SecondOrderResidual>(pfes_v, rho);
  FirstOrderSystem          the_first_order(residual);
  mfem::BackwardEulerSolver be;
  be.Init(the_first_order);

  mfem::Array<int> offsets(3);
  offsets[0] = 0;
  offsets[1] = offsets[0] + pfes_v->GetVSize();
  offsets[2] = offsets[1] + pfes_v->GetVSize();

  // u(0) = no displacements
  mfem::BlockVector     x0(offsets);
  mfem::ParGridFunction u0(pfes_v.get());
  u0 = 0.;

  mfem::ParGridFunction v0(pfes_v.get());
  v0 = 0.;

  x0.GetBlock(0) = u0;
  x0.GetBlock(1) = v0;

  // Domain is not accelerating at t=0
  x0.GetBlock(2) = 0.;

  // Fix x = 0
  int                           ne = nex;
  serac::StdFunctionCoefficient fixed([ne](mfem::Vector& x) { return (x[0] < 1. / ne) ? 1. : 0.; });

  mfem::Array<int> bdr_attr_list = serac::makeBdrAttributeList(*pmesh, fixed);
  for (int be = 0; be < pmesh->GetNBE(); be++) {
    pmesh->GetBdrElement(be)->SetAttribute(bdr_attr_list[be]);
  }
  pmesh->SetAttributes();

  mfem::Array<int> ess_tdof_list;
  mfem::Array<int> bdr_attr_is_ess(pmesh->bdr_attributes.Max());
  bdr_attr_is_ess = 0;
  if (bdr_attr_is_ess.Size() > 1) bdr_attr_is_ess[1] = 1;
  pfes_v->GetEssentialTrueDofs(bdr_attr_is_ess, ess_tdof_list);

  // Homogeneous dirchlet bc
  mfem::Vector u_ess(ess_tdof_list.Size());
  u_ess = 0.;

  residual->setBoundaryConditions(ess_tdof_list, u_ess);

  // Add self-weight
  mfem::Vector grav(dim);
  grav    = 0.;
  grav[1] = g;
  mfem::VectorConstantCoefficient grav_v_coef(grav);
  // Here rho is the actually rho = rho_const * vf
  mfem::ScalarVectorProductCoefficient gravity_load_coef(rho, grav_v_coef);
  auto                                 gravity2 = std::make_unique<mfem::VectorDomainLFIntegrator>(gravity_load_coef);
  residual->addLinearDomainIntegrator(std::move(gravity2));

  // Make it non rigid body
  mfem::ConstantCoefficient lambda(lambda_c);
  mfem::ConstantCoefficient mu(mu_c);
  auto                      elasticity2 = std::make_unique<mfem::ElasticityIntegrator>(lambda, mu);
  residual->addBilinearDomainIntegrator(std::move(elasticity2));

  cout << "output volume fractions :" << endl;

  mfem::ParGridFunction gravity_eval(pfes_v.get());
  gravity_eval.ProjectCoefficient(gravity_load_coef);
  gravity_eval.Print();

  double t = 0.;

  std::cout << "t = " << t << std::endl;

  // Store results
  mfem::Vector u_next(x0.GetBlock(0));
  mfem::Vector v_next(x0.GetBlock(1));

  mfem::ParGridFunction u_visit(pfes_v.get());
  mfem::ParGridFunction v_visit(pfes_v.get());
  u_visit = u_next;
  v_visit = v_next;

  mfem::VisItDataCollection visit("FirstOrder", pmesh.get());
  visit.RegisterField("u_next", &u_visit);
  visit.RegisterField("v_next", &v_visit);
  visit.SetCycle(0);
  visit.SetTime(0.);
  visit.Save();

  for (int i = 0; i < num_time_steps; i++) {
    mfem::Vector x0_temp(x0.GetData(), offsets[2]);
    be.Step(x0_temp, t, dt);

    u_visit = x0.GetBlock(0);
    v_visit = x0.GetBlock(1);
    visit.SetCycle(i + 1);
    visit.SetTime(t);
    visit.Save();
  }
}
