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

class NewmarkBetaSecondOrder : public mfem::SecondOrderTimeDependentOperator {
public:
  NewmarkBetaSecondOrder(std::shared_ptr<mfem::ParFiniteElementSpace> pfes, mfem::Coefficient& density, double beta,
                         double gamma)
      : beta_(beta),
        gamma_(gamma),
        pfes_(pfes),
        density_(density),
        offsets_(4),
        up_(new mfem::ParGridFunction(pfes.get()))
  {
    *up_ = 0.;
    // Create offsets to parse the input vector as a blockvector
    offsets_[0] = 0;
    offsets_[1] = offsets_[0] + pfes_->GetVSize();
    offsets_[2] = offsets_[1] + pfes_->GetVSize();
    offsets_[3] = offsets_[2] + pfes_->GetVSize();

    mfem::Vector ones(pfes_->GetVDim());
    ones           = 1.;
    auto ones_coef          = std::make_shared<mfem::VectorConstantCoefficient>(ones);
    auto inertial_coef = std::make_shared<mfem::ScalarVectorProductCoefficient>(density_, *ones_coef);

    auto inertial_integrator = std::make_shared<mfem::VectorMassIntegrator>(density_);
    auto nonlinear_inertial_integrator =
        std::make_shared<serac::BilinearToNonlinearFormIntegrator>(inertial_integrator);
    residual_int_.push_back(nonlinear_inertial_integrator);
  }

  void AddBilinearDomainIntegrator(std::unique_ptr<mfem::BilinearFormIntegrator> blfi)
  {
    auto shared_version = std::shared_ptr<mfem::BilinearFormIntegrator>(blfi.release());
    residual_int_u_.push_back(std::make_shared<serac::BilinearToNonlinearFormIntegrator>(shared_version));
  }

  void AddLinearDomainIntegrator(std::unique_ptr<mfem::LinearFormIntegrator> lfi)
  {
    auto shared_version = std::shared_ptr<mfem::LinearFormIntegrator>(lfi.release());
    residual_int_.push_back(std::make_shared<serac::LinearToNonlinearFormIntegrator>(shared_version, pfes_));
  }

  void AddNonlinearDomainIntegrator(std::unique_ptr<mfem::NonlinearFormIntegrator> nlfi)
  {
    auto shared_version = std::shared_ptr<mfem::NonlinearFormIntegrator>(nlfi.release());
    residual_int_.push_back(shared_version);
  }

  // Set Essential Boundary Conditions
  void SetBoundaryConditions(mfem::Array<int> local_tdofs, mfem::Vector vals)
  {
    MFEM_VERIFY(local_tdofs.Size() == vals.Size(), "Essential true dof size != val.Size()");

    *up_ = 0.;
    up_->SetSubVector(local_tdofs, vals);

    local_ess_tdofs_ = local_tdofs;
  }

  void SolveResidual(mfem::ParNonlinearForm& R, mfem::ParGridFunction& sol_next) const
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

  /// NewmarkBeta uses this to get the initial acceleration
  // It's assumed that x, dxdt, and y have the same size
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

    mfem::ParNonlinearForm R(pfes_.get());
    for (auto integ : residual_int_) R.AddDomainIntegrator(new serac::PointerNonlinearFormIntegrator(integ));

    for (auto integ : residual_int_u_) {
      auto sub_non_integ =
          std::make_shared<serac::SubstitutionNonlinearFormIntegrator>(integ, substitute_u0, zero_grad);
      R.AddDomainIntegrator(new serac::PointerNonlinearFormIntegrator(sub_non_integ));
    }

    a0 = 0.;
    SolveResidual(R, a0);
  }

  virtual void ImplicitSolve(const double, const double dt1, const mfem::Vector& x, const mfem::Vector& dxdt,
                             mfem::Vector& k) override
  {
    // x is already x_pred
    // dxdt is already dxdt_pred
    double dt = dt1 / gamma_;

    mfem::ParGridFunction u_pred(pfes_.get());
    u_pred.SetFromTrueDofs(x);
    mfem::ParGridFunction v_pred(pfes_.get());
    v_pred.SetFromTrueDofs(dxdt);
    mfem::ParGridFunction a_next(pfes_.get());
    a_next.SetFromTrueDofs(k);
    a_next = 0.;
    mfem::ParGridFunction u_next(pfes_.get());

    // Currently u_next = u_pred, v_next = v_pred
    // u_next(a_next) = u_pred + beta * dt * dt   * a_next
    // v_next(a_next) = v_pred + gamma * dt * a_next
    // a_next(u_next) = (u_next - u_pred)/(beta * dt * dt)

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

    // du_next(a_next)/da_next = beta * dt * dt
    auto substitute_a_next_grad = [&](const mfem::FiniteElement&, mfem::ElementTransformation&,
                                      const mfem::DenseMatrix& elmat) {
      auto m = std::make_shared<mfem::DenseMatrix>(elmat);
      *m *= dadu;
      return m;
    };

    mfem::ParNonlinearForm R(pfes_.get());

    for (auto integ : residual_int_) {
      auto sub_non_integ = std::make_shared<serac::SubstitutionNonlinearFormIntegrator>(integ, substitute_a_next,
                                                                                        substitute_a_next_grad);

      R.AddDomainIntegrator(new serac::PointerNonlinearFormIntegrator(sub_non_integ));
    }

    for (auto integ : residual_int_u_) {
      R.AddDomainIntegrator(new serac::PointerNonlinearFormIntegrator(integ));
    }

    R.SetEssentialTrueDofs(local_ess_tdofs_);

    u_next = *up_;

    SolveResidual(R, u_next);
    evaluate(dadu * (u_next - u_pred), a_next);

    a_next.GetTrueDofs(k);
  }

  std::shared_ptr<mfem::ParFiniteElementSpace> GetParFESpace() { return pfes_; }


  const mfem::Array<int>& GetLocalTDofs() { return local_ess_tdofs_; }

  std::vector<std::shared_ptr<mfem::NonlinearFormIntegrator>> residual_int_u_;  //< dependent on u
  std::vector<std::shared_ptr<mfem::NonlinearFormIntegrator>> residual_int_;    //< not dependent on u or v

  std::unique_ptr<mfem::ParGridFunction> up_;  //< holds essential boundary conditions

private:
  mfem::Array<int> local_ess_tdofs_;  //< local true essential boundary conditions

  double beta_, gamma_;  //< Newmark parameters

  std::shared_ptr<mfem::ParFiniteElementSpace> pfes_;  //< finite element space

  mfem::Coefficient& density_;  //< density
  mfem::Array<int>   offsets_;  //< local block offsets

};

TEST_F(NewmarkBetaTest, Simple)
{
  double beta  = 0.25;
  double gamma = 0.5;

  mfem::ConstantCoefficient rho(1.0);

  NewmarkBetaSecondOrder second_order(pfes_v, rho, beta, gamma);
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
  NewmarkBetaSecondOrder    second_order(pfes_v, rho, beta, gamma);
  mfem::NewmarkSolver       ns(beta, gamma);
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

  second_order.SetBoundaryConditions(ess_tdof_list, u_ess);

  // Add self-weight
  mfem::Vector grav(dim);
  grav    = 0.;
  grav[1] = g;
  mfem::VectorConstantCoefficient grav_v_coef(grav);
  // Here rho is the actually rho = rho_const * vf
  mfem::ScalarVectorProductCoefficient gravity_load_coef(rho, grav_v_coef);
  auto                                 gravity2 = std::make_unique<mfem::VectorDomainLFIntegrator>(gravity_load_coef);
  second_order.AddLinearDomainIntegrator(std::move(gravity2));

  // Make it non rigid body
  mfem::ConstantCoefficient lambda(lambda_c);
  mfem::ConstantCoefficient mu(mu_c);
  auto                      elasticity2 = std::make_unique<mfem::ElasticityIntegrator>(lambda, mu);
  second_order.AddBilinearDomainIntegrator(std::move(elasticity2));

  cout << "output volume fractions :" << endl;

  mfem::ParGridFunction gravity_eval(pfes_v.get());
  gravity_eval.ProjectCoefficient(gravity_load_coef);
  gravity_eval.Print();

  double t = 0.;

  std::cout << "t = " << t << std::endl;

  // Store results
  mfem::Vector u_prev(x0.GetBlock(0));
  mfem::Vector v_prev(x0.GetBlock(1));
  mfem::Vector a_prev(x0.GetBlock(2));

  mfem::Vector u_next(x0.GetBlock(0));
  mfem::Vector v_next(x0.GetBlock(1));
  mfem::Vector a_next(x0.GetBlock(2));

  mfem::ParGridFunction u_visit(pfes_v.get());
  u_visit = u_next;
  mfem::ParGridFunction v_visit(pfes_v.get());
  v_visit = v_next;
  mfem::ParGridFunction a_visit(pfes_v.get());
  a_visit = a_next;

  mfem::VisItDataCollection visit("NewmarkBeta", pmesh.get());
  visit.RegisterField("u_next", &u_visit);
  visit.RegisterField("v_next", &v_visit);
  visit.RegisterField("a_next", &a_visit);
  visit.SetCycle(0);
  visit.SetTime(0.);
  visit.Save();

  for (int i = 0; i < num_time_steps; i++) {
    ns.Step(x0.GetBlock(0), x0.GetBlock(1), t, dt);
    u_visit = x0.GetBlock(0);
    a_visit = x0.GetBlock(2);
    visit.SetTime(t);
    visit.SetCycle(i + 1);
    visit.Save();
  }
}

class FirstOrderSystem : public mfem::TimeDependentOperator {
public:
  FirstOrderSystem(std::shared_ptr<NewmarkBetaSecondOrder> tdo)
      : mfem::TimeDependentOperator(tdo->GetParFESpace()->GetTrueVSize() * 2, 0.,
                                    mfem::TimeDependentOperator::Type::IMPLICIT),
        tdo_(tdo),
        offsets_(3)
  {
    offsets_[0] = 0;
    offsets_[1] = offsets_[0] + tdo_->GetParFESpace()->GetTrueVSize();
    offsets_[2] = offsets_[1] + tdo_->GetParFESpace()->GetTrueVSize();
  }

  virtual void Mult(const mfem::Vector&, mfem::Vector&) const
  {
    mfem::mfem_error("not currently supported for this second order wrapper");
  };

  virtual void ImplicitSolve(const double dt, const mfem::Vector& x, mfem::Vector& k) override
  {
    MFEM_VERIFY(x.Size() == offsets_[2], "vectors are not the same size");
    mfem::BlockVector bx(x.GetData(), offsets_);

    /*
      NewmarkBetaSecondOrder has a SolveResidual routine that currently updates u(next(a) based on an internal update
      function

      u_next = u_prev + dt * v_next
      v_next = v_prev + dt * a_next
      u_next = u_prev + dt * (v_prev + dt * a_next);
      u_next = u_prev + dt * v_prev + dt*dt*a_next
      a_next(u_next) = (u_next - u_prev - dt * v_prev)/(dt*dt)

    */

    mfem::ParGridFunction u_prev(tdo_->GetParFESpace().get());
    mfem::ParGridFunction v_prev(tdo_->GetParFESpace().get());
    u_prev.SetFromTrueDofs(bx.GetBlock(0));
    v_prev.SetFromTrueDofs(bx.GetBlock(1));

    mfem::ParGridFunction a_next(tdo_->GetParFESpace().get());
    a_next = 0.;
    mfem::ParGridFunction u_next(tdo_->GetParFESpace().get());

    auto substitute_a = [&](const mfem::FiniteElement&, mfem::ElementTransformation& Tr, const mfem::Vector& u_next_elem) {
      auto                         a_next_elem = std::make_shared<mfem::Vector>(u_next_elem.Size());
      int                          e      = Tr.ElementNo;
      mfem::ParFiniteElementSpace* pfes   = u_prev.ParFESpace();
      mfem::Array<int>             vdofs;
      pfes->GetElementVDofs(e, vdofs);
      mfem::Vector u_prev_elem(u_prev.Size());
      u_prev.GetSubVector(vdofs, u_prev_elem);
      mfem::Vector v_prev_elem(u_prev.Size());
      v_prev.GetSubVector(vdofs, v_prev_elem);

      *a_next_elem = (u_next_elem - u_prev_elem - v_prev_elem * dt) / (dt * dt);

      return a_next_elem;
    };

    double dadu              = 1. / (dt * dt);
    auto   substitute_a_grad = [&](const mfem::FiniteElement&, mfem::ElementTransformation&,
                                 const mfem::DenseMatrix& elmat) {
      auto m = std::make_shared<mfem::DenseMatrix>(elmat);
      *m *= dadu;
      return m;
    };

    mfem::ParNonlinearForm R(tdo_->GetParFESpace().get());

    for (auto integ : tdo_->residual_int_) {
      auto sub_non_integ =
          std::make_shared<serac::SubstitutionNonlinearFormIntegrator>(integ, substitute_a, substitute_a_grad);

      R.AddDomainIntegrator(new serac::PointerNonlinearFormIntegrator(sub_non_integ));
    }

    for (auto integ : tdo_->residual_int_u_) {
      R.AddDomainIntegrator(new serac::PointerNonlinearFormIntegrator(integ));
    }

    R.SetEssentialTrueDofs(tdo_->GetLocalTDofs());

    u_next = *tdo_->up_;
    tdo_->SolveResidual(R, u_next);
    evaluate(1. / (dt * dt) * (u_next - u_prev - dt * v_prev), a_next);

    // v_next = v_old + dt * a_next
    mfem::ParGridFunction v_next(tdo_->GetParFESpace().get());
    v_next = v_prev + dt * a_next;

    // set the results back
    mfem::BlockVector bk(k.GetData(), offsets_);
    v_next.GetTrueDofs(bk.GetBlock(0));
    a_next.GetTrueDofs(bk.GetBlock(1));
  }

protected:
  std::shared_ptr<NewmarkBetaSecondOrder> tdo_;
  mfem::Array<int>                        offsets_;
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
  auto                      second_order = std::make_shared<NewmarkBetaSecondOrder>(pfes_v, rho, beta, gamma);
  FirstOrderSystem          the_first_order(second_order);
  mfem::BackwardEulerSolver be;
  be.Init(the_first_order);

  mfem::NewmarkSolver ns(beta, gamma);
  ns.Init(*second_order);

  mfem::Array<int> offsets(4);
  offsets[0] = 0;
  offsets[1] = offsets[0] + pfes_v->GetVSize();
  offsets[2] = offsets[1] + pfes_v->GetVSize();
  offsets[3] = offsets[2] + pfes_v->GetVSize();

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

  second_order->SetBoundaryConditions(ess_tdof_list, u_ess);

  // Add self-weight
  mfem::Vector grav(dim);
  grav    = 0.;
  grav[1] = g;
  mfem::VectorConstantCoefficient grav_v_coef(grav);
  // Here rho is the actually rho = rho_const * vf
  mfem::ScalarVectorProductCoefficient gravity_load_coef(rho, grav_v_coef);
  auto                                 gravity2 = std::make_unique<mfem::VectorDomainLFIntegrator>(gravity_load_coef);
  second_order->AddLinearDomainIntegrator(std::move(gravity2));

  // Make it non rigid body
  mfem::ConstantCoefficient lambda(lambda_c);
  mfem::ConstantCoefficient mu(mu_c);
  auto                      elasticity2 = std::make_unique<mfem::ElasticityIntegrator>(lambda, mu);
  second_order->AddBilinearDomainIntegrator(std::move(elasticity2));

  cout << "output volume fractions :" << endl;

  mfem::ParGridFunction gravity_eval(pfes_v.get());
  gravity_eval.ProjectCoefficient(gravity_load_coef);
  gravity_eval.Print();

  double t = 0.;

  std::cout << "t = " << t << std::endl;

  // Store results
  mfem::Vector u_prev(x0.GetBlock(0));
  mfem::Vector v_prev(x0.GetBlock(1));
  mfem::Vector a_prev(x0.GetBlock(2));

  mfem::Vector u_next(x0.GetBlock(0));
  mfem::Vector v_next(x0.GetBlock(1));
  mfem::Vector a_next(x0.GetBlock(2));

  mfem::ParGridFunction u_visit(pfes_v.get());
  u_visit = u_next;
  mfem::ParGridFunction v_visit(pfes_v.get());
  v_visit = v_next;
  mfem::ParGridFunction a_visit(pfes_v.get());
  a_visit = a_next;

  mfem::VisItDataCollection visit("FirstOrder", pmesh.get());
  visit.RegisterField("u_next", &u_visit);
  visit.RegisterField("v_next", &v_visit);
  visit.SetCycle(0);
  visit.SetTime(0.);
  visit.Save();

  mfem::BlockVector x0_copy(x0);

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
