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

using mfem::Array;
using mfem::BasisType;
using mfem::BilinearFormIntegrator;
using mfem::BlockVector;
using mfem::Coefficient;
using mfem::ConstantCoefficient;
using mfem::ElementTransformation;
using mfem::HypreBoomerAMG;
using mfem::HypreParMatrix;
using mfem::HypreParVector;
using mfem::HyprePCG;
using mfem::IntegrationPoint;
using mfem::LinearFormIntegrator;
using mfem::Mesh;
using mfem::Ordering;
using mfem::ParBilinearForm;
using mfem::ParFiniteElementSpace;
using mfem::ParGridFunction;
using mfem::ParLinearForm;
using mfem::ParMesh;
using mfem::ScalarVectorProductCoefficient;
using mfem::SparseMatrix;
using mfem::TimeDependentOperator;
using mfem::Vector;
using mfem::VectorConstantCoefficient;
using mfem::VectorGridFunctionCoefficient;
using mfem::VectorMassIntegrator;

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

    Mesh mesh(nex, ney, mfem::Element::QUADRILATERAL, 1, len, width);
    pmesh  = std::shared_ptr<ParMesh>(new ParMesh(MPI_COMM_WORLD, mesh));
    pfes   = std::shared_ptr<ParFiniteElementSpace>(new ParFiniteElementSpace(
        pmesh.get(), new mfem::H1_FECollection(1, dim, BasisType::GaussLobatto), 1, Ordering::byNODES));
    pfes_v = std::shared_ptr<ParFiniteElementSpace>(new ParFiniteElementSpace(
        pmesh.get(), new mfem::H1_FECollection(1, dim, BasisType::GaussLobatto), dim, Ordering::byNODES));

    pfes_l2 = std::shared_ptr<ParFiniteElementSpace>(
        new ParFiniteElementSpace(pmesh.get(), new mfem::L2_FECollection(0, dim), 1, Ordering::byNODES));
  }

  void TearDown() {}

  double                                 width, len;
  int                                    nex, ney, nez;
  int                                    dim;
  std::shared_ptr<ParMesh>               pmesh;
  std::shared_ptr<ParFiniteElementSpace> pfes;
  std::shared_ptr<ParFiniteElementSpace> pfes_v;
  std::shared_ptr<ParFiniteElementSpace> pfes_l2;
};

class TransformedCoefficient : public Coefficient {
public:
  TransformedCoefficient(Coefficient& C, std::function<double(double)> func) : C_(&C), func_(func) {}

  virtual ~TransformedCoefficient() {}

  virtual double Eval(ElementTransformation& Tr, const IntegrationPoint& ip) { return func_(C_->Eval(Tr, ip)); }

private:
  Coefficient*                  C_;
  std::function<double(double)> func_;
};

class NewmarkBetaSecondOrder : public mfem::SecondOrderTimeDependentOperator {
public:
  NewmarkBetaSecondOrder(std::shared_ptr<mfem::ParFiniteElementSpace> pfes, mfem::Coefficient& density, double beta,
                         double gamma)
      : beta_(beta), gamma_(gamma), pfes_(pfes), density_(&density), offsets_(4), up_(new ParGridFunction(pfes.get()))
  {
    *up_ = 0.;
    // Create offsets to parse the input vector as a blockvector
    offsets_[0] = 0;
    offsets_[1] = offsets_[0] + pfes_->GetVSize();
    offsets_[2] = offsets_[1] + pfes_->GetVSize();
    offsets_[3] = offsets_[2] + pfes_->GetVSize();

    Vector ones(pfes_->GetVDim());
    ones           = 1.;
    ones_          = std::make_shared<VectorConstantCoefficient>(ones);
    inertial_coef_ = std::make_shared<mfem::ScalarVectorProductCoefficient>(*density_, *ones_);

    auto inertial_integrator = std::make_shared<VectorMassIntegrator>(*inertial_coef_);
    auto nonlinear_inertial_integrator =
        std::make_shared<serac::BilinearToNonlinearFormIntegrator>(inertial_integrator);
    R_int_.push_back(nonlinear_inertial_integrator);
  }

  void AddBilinearDomainIntegrator(std::unique_ptr<BilinearFormIntegrator> blfi)
  {
    auto shared_version = std::shared_ptr<mfem::BilinearFormIntegrator>(blfi.release());
    R_int_u_.push_back(std::make_shared<serac::BilinearToNonlinearFormIntegrator>(shared_version));
  }

  void AddLinearDomainIntegrator(std::unique_ptr<LinearFormIntegrator> lfi)
  {
    auto shared_version = std::shared_ptr<mfem::LinearFormIntegrator>(lfi.release());
    R_int_.push_back(std::make_shared<serac::LinearToNonlinearFormIntegrator>(shared_version, pfes_));
  }

  void AddNonlinearDomainIntegrator(std::unique_ptr<mfem::NonlinearFormIntegrator> nlfi)
  {
    auto shared_version = std::shared_ptr<mfem::NonlinearFormIntegrator>(nlfi.release());
    R_int_.push_back(shared_version);
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

    auto Sol_next = std::unique_ptr<HypreParVector>(sol_next.GetTrueDofs());

    Vector zero;
    newton_solver.Mult(zero, *Sol_next);

    int num_iterations_taken = newton_solver.GetNumIterations();
    std::cout << "initial iterations:" << num_iterations_taken << std::endl;
    // Copy solution back
    sol_next = *Sol_next;
  }

  /// NewmarkBeta uses this to get the initial acceleration
  // It's assumed that x, dxdt, and y have the same size
  virtual void Mult(const Vector& x, const Vector& dxdt, Vector& y) const override
  {
    // convert x, dxdt, and y into ParGridFunctions
    ParGridFunction u0(pfes_.get(), x.GetData());
    ParGridFunction v0(pfes_.get(), dxdt.GetData());
    y.SetSize(u0.Size());
    ParGridFunction a0(pfes_.get(), y.GetData());

    auto zero_grad = [](const mfem::FiniteElement&, mfem::ElementTransformation&, const mfem::DenseMatrix& elmat) {
      auto m = std::make_shared<mfem::DenseMatrix>(elmat);
      *m     = 0.;
      return m;
    };

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
    for (auto integ : R_int_) R.AddDomainIntegrator(new serac::PointerNonlinearFormIntegrator(integ));

    for (auto integ : R_int_u_) {
      auto sub_non_integ =
          std::make_shared<serac::SubstitutionNonlinearFormIntegrator>(integ, substitute_u0, zero_grad);
      R.AddDomainIntegrator(new serac::PointerNonlinearFormIntegrator(sub_non_integ));
    }

    a0 = 0.;
    SolveResidual(R, a0);
  }

  virtual void ImplicitSolve(const double, const double dt1, const Vector& x, const Vector& dxdt, Vector& k) override
  {
    // x is already x_pred
    // dxdt is already dxdt_pred
    double dt = dt1 / gamma_;

    ParGridFunction u_pred(pfes_.get());
    u_pred.SetFromTrueDofs(x);
    ParGridFunction v_pred(pfes_.get());
    v_pred.SetFromTrueDofs(dxdt);
    ParGridFunction a_next(pfes_.get());
    a_next.SetFromTrueDofs(k);
    a_next = 0.;
    ParGridFunction u_next(pfes_.get());

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
      Vector u_pred_vect(u_next.Size());
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

    for (auto integ : R_int_) {
      auto sub_non_integ = std::make_shared<serac::SubstitutionNonlinearFormIntegrator>(integ, substitute_a_next,
                                                                                        substitute_a_next_grad);

      R.AddDomainIntegrator(new serac::PointerNonlinearFormIntegrator(sub_non_integ));
    }

    for (auto integ : R_int_u_) {
      R.AddDomainIntegrator(new serac::PointerNonlinearFormIntegrator(integ));
    }

    R.SetEssentialTrueDofs(local_ess_tdofs_);

    u_next = *up_;

    SolveResidual(R, u_next);
    evaluate(dadu * (u_next - u_pred), a_next);

    a_next.GetTrueDofs(k);
  }

  std::shared_ptr<mfem::ParFiniteElementSpace> GetParFESpace() { return pfes_; }

  serac::SubstitutionNonlinearFormIntegrator::SubstituteFunction SwapEval(
      serac::SubstitutionNonlinearFormIntegrator::SubstituteFunction func)
  {
    auto temp = SubEval_;
    SubEval_  = func;
    return temp;
  }

  serac::SubstitutionNonlinearFormIntegrator::SubstituteGradFunction SwapGradEval(
      serac::SubstitutionNonlinearFormIntegrator::SubstituteGradFunction func)
  {
    auto temp    = SubGradEval_;
    SubGradEval_ = func;
    return temp;
  }

  const mfem::Array<int>& GetLocalTDofs() { return local_ess_tdofs_; }

  std::vector<std::shared_ptr<mfem::NonlinearFormIntegrator>> R_int_u_;  //< dependent on u
  std::vector<std::shared_ptr<mfem::NonlinearFormIntegrator>> R_int_;    //< not dependent on u or v

private:
  mfem::Array<int> local_ess_tdofs_;  //< local true essential boundary conditions

  double beta_, gamma_;  //< Newmark parameters

  std::shared_ptr<mfem::ParFiniteElementSpace> pfes_;  //< finite element space

  mfem::Coefficient* density_;  //< density
  mfem::Array<int>   offsets_;  //< local block offsets

  std::shared_ptr<mfem::VectorCoefficient> inertial_coef_;  //< holds the vector inertial mass integrator
  std::shared_ptr<mfem::VectorCoefficient> ones_;           //< a vector of ones

  std::unique_ptr<ParGridFunction> up_;  //< holds essential boundary conditions

  // change of variable functions
  mutable serac::SubstitutionNonlinearFormIntegrator::SubstituteFunction
      SubEval_;  //< method to perform change of variables on u(a)
  mutable serac::SubstitutionNonlinearFormIntegrator::SubstituteGradFunction
      SubGradEval_;  //< method to perform change of variables on u_grad(a)
};

TEST_F(NewmarkBetaTest, Simple)
{
  double beta  = 0.25;
  double gamma = 0.5;

  ConstantCoefficient rho(1.0);

  NewmarkBetaSecondOrder second_order(pfes_v, rho, beta, gamma);
  mfem::NewmarkSolver    ns(beta, gamma);
  ns.Init(second_order);

  mfem::Array<int> offsets(4);
  offsets[0] = 0;
  offsets[1] = offsets[0] + pfes_v->GetVSize();
  offsets[2] = offsets[1] + pfes_v->GetVSize();
  offsets[3] = offsets[2] + pfes_v->GetVSize();

  // u(0) = no displacements
  BlockVector     x0(offsets);
  ParGridFunction u0(pfes_v.get());
  u0 = 0.;

  // v(0) = 1 m/s in the y direction
  ParGridFunction v0(pfes_v.get());
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
  Vector u_prev(x0.GetBlock(0));
  Vector v_prev(x0.GetBlock(1));
  Vector a_prev(x0.GetBlock(1));

  Vector u_next(u_prev);
  Vector v_next(v_prev);
  second_order.Mult(u_prev, v_prev, a_prev);

  ns.Step(u_next, v_next, t, dt);

  std::cout << "t = " << t << std::endl;
  std::cout << "x" << std::endl;
  u_next.Print();
  std::cout << "v" << std::endl;
  v_next.Print();

  // back out a_next
  Vector a_next(v_next);
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
  using VisItDataCollection = mfem::VisItDataCollection;

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

  double beta  = 0.25;
  double gamma = 0.5;

  ConstantCoefficient    rho(r);
  NewmarkBetaSecondOrder second_order(pfes_v, rho, beta, gamma);
  mfem::NewmarkSolver    ns(beta, gamma);
  ns.Init(second_order);

  mfem::Array<int> offsets(4);
  offsets[0] = 0;
  offsets[1] = offsets[0] + pfes_v->GetVSize();
  offsets[2] = offsets[1] + pfes_v->GetVSize();
  offsets[3] = offsets[2] + pfes_v->GetVSize();

  // u(0) = no displacements
  BlockVector     x0(offsets);
  ParGridFunction u0(pfes_v.get());
  u0 = 0.;

  ParGridFunction v0(pfes_v.get());
  v0 = 0.;

  x0.GetBlock(0) = u0;
  x0.GetBlock(1) = v0;

  // Domain is not accelerating at t=0
  x0.GetBlock(2) = 0.;

  // Fix x = 0
  int                           ne = nex;
  serac::StdFunctionCoefficient fixed([ne](Vector& x) { return (x[0] < 1. / ne) ? 1. : 0.; });

  mfem::Array<int> bdr_attr_list = serac::makeBdrAttributeList(*pmesh, fixed);
  for (int be = 0; be < pmesh->GetNBE(); be++) {
    pmesh->GetBdrElement(be)->SetAttribute(bdr_attr_list[be]);
  }
  pmesh->SetAttributes();

  Array<int> ess_tdof_list;
  Array<int> bdr_attr_is_ess(pmesh->bdr_attributes.Max());
  bdr_attr_is_ess = 0;
  if (bdr_attr_is_ess.Size() > 1) bdr_attr_is_ess[1] = 1;
  pfes_v->GetEssentialTrueDofs(bdr_attr_is_ess, ess_tdof_list);

  // Homogeneous dirchlet bc
  Vector u_ess(ess_tdof_list.Size());
  u_ess = 0.;

  second_order.SetBoundaryConditions(ess_tdof_list, u_ess);

  // Add self-weight
  Vector grav(dim);
  grav    = 0.;
  grav[1] = g;
  VectorConstantCoefficient grav_v_coef(grav);
  // Here rho is the actually rho = rho_const * vf
  ScalarVectorProductCoefficient gravity_load_coef(rho, grav_v_coef);
  auto                           gravity2 = std::make_unique<mfem::VectorDomainLFIntegrator>(gravity_load_coef);
  second_order.AddLinearDomainIntegrator(std::move(gravity2));

  // Make it non rigid body
  ConstantCoefficient lambda(lambda_c);
  ConstantCoefficient mu(mu_c);
  auto                elasticity2 = std::make_unique<mfem::ElasticityIntegrator>(lambda, mu);
  second_order.AddBilinearDomainIntegrator(std::move(elasticity2));

  cout << "output volume fractions :" << endl;

  ParGridFunction gravity_eval(pfes_v.get());
  gravity_eval.ProjectCoefficient(gravity_load_coef);
  gravity_eval.Print();

  double t = 0.;

  std::cout << "t = " << t << std::endl;

  // Store results
  Vector u_prev(x0.GetBlock(0));
  Vector v_prev(x0.GetBlock(1));
  Vector a_prev(x0.GetBlock(2));

  Vector u_next(x0.GetBlock(0));
  Vector v_next(x0.GetBlock(1));
  Vector a_next(x0.GetBlock(2));

  ParGridFunction u_visit(pfes_v.get());
  u_visit = u_next;
  ParGridFunction v_visit(pfes_v.get());
  v_visit = v_next;
  ParGridFunction a_visit(pfes_v.get());
  a_visit = a_next;

  VisItDataCollection visit("NewmarkBeta", pmesh.get());
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

  virtual void Mult(const Vector&, Vector&) const
  {
    mfem::mfem_error("not currently supported for this second order wrapper");
  };

  virtual void ImplicitSolve(const double dt, const Vector& x, Vector& k) override
  {
    MFEM_VERIFY(x.Size() == offsets_[2], "vectors are not the same size");
    mfem::BlockVector bx(x.GetData(), offsets_);

    /*
      NewmarkBetaSecondOrder has a SolveResidual routine that currently updates u(next(a) based on an internal update
      function

      du/dt = v_next

      u_next = u_old + dt * v_next
      v_next = v_old + dt * a_next
      u_next = u_old + dt * (v_old + dt * a_next)
      u_next = u_old + dt * v_old + dt * dt * a_next)
      du_next/da_next = dt * dt
      a_next = (u_next - u_old - dt * v_old) / dt^2
      da_next/du_next = 1 / dt^2

      M*a_next(u_next) + K*u_next + F_ext = 0
     */

    mfem::ParGridFunction u_prev(tdo_->GetParFESpace().get());
    mfem::ParGridFunction v_prev(tdo_->GetParFESpace().get());
    u_prev.SetFromTrueDofs(bx.GetBlock(0));
    v_prev.SetFromTrueDofs(bx.GetBlock(1));

    mfem::ParGridFunction a_next(tdo_->GetParFESpace().get());
    a_next = 0.;

    auto substitute_a = [&](const mfem::FiniteElement&, mfem::ElementTransformation& Tr, const mfem::Vector& u_next) {
      auto                         a_next = std::make_shared<mfem::Vector>(u_next.Size());
      int                          e      = Tr.ElementNo;
      mfem::ParFiniteElementSpace* pfes   = u_prev.ParFESpace();
      mfem::Array<int>             vdofs;
      pfes->GetElementVDofs(e, vdofs);
      mfem::Vector u_elem(u_next.Size());
      u_prev.GetSubVector(vdofs, u_elem);
      mfem::Vector v_elem(u_next.Size());
      v_prev.GetSubVector(vdofs, v_elem);

      *a_next = (u_next - u_elem - v_elem * dt) / (dt * dt);

      return a_next;
    };

    auto substitute_a_grad = [&](const mfem::FiniteElement&, mfem::ElementTransformation&,
                                 const mfem::DenseMatrix& elmat) {
      auto m = std::make_shared<mfem::DenseMatrix>(elmat);
      *m *= 1. / (dt * dt);
      return m;
    };

    mfem::ParNonlinearForm R(tdo_->GetParFESpace().get());

    for (auto integ : tdo_->R_int_) {
      auto sub_non_integ =
          std::make_shared<serac::SubstitutionNonlinearFormIntegrator>(integ, substitute_a, substitute_a_grad);

      R.AddDomainIntegrator(new serac::PointerNonlinearFormIntegrator(sub_non_integ));
    }

    for (auto integ : tdo_->R_int_u_) {
      R.AddDomainIntegrator(new serac::PointerNonlinearFormIntegrator(integ));
    }

    R.SetEssentialTrueDofs(tdo_->GetLocalTDofs());

    tdo_->SolveResidual(R, a_next);

    // v_next = v_old + dt * a_next
    ParGridFunction v_next(tdo_->GetParFESpace().get());
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
  using VisItDataCollection = mfem::VisItDataCollection;

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

  double beta  = 0.25;
  double gamma = 0.5;

  ConstantCoefficient       rho(r);
  auto                      second_order = std::make_shared<NewmarkBetaSecondOrder>(pfes_v, rho, beta, gamma);
  FirstOrderSystem          the_first_order(second_order);
  mfem::BackwardEulerSolver be;
  be.Init(the_first_order);

  mfem::Array<int> offsets(4);
  offsets[0] = 0;
  offsets[1] = offsets[0] + pfes_v->GetVSize();
  offsets[2] = offsets[1] + pfes_v->GetVSize();
  offsets[3] = offsets[2] + pfes_v->GetVSize();

  // u(0) = no displacements
  BlockVector     x0(offsets);
  ParGridFunction u0(pfes_v.get());
  u0 = 0.;

  ParGridFunction v0(pfes_v.get());
  v0 = 0.;

  x0.GetBlock(0) = u0;
  x0.GetBlock(1) = v0;

  // Domain is not accelerating at t=0
  x0.GetBlock(2) = 0.;

  // Fix x = 0
  int                           ne = nex;
  serac::StdFunctionCoefficient fixed([ne](Vector& x) { return (x[0] < 1. / ne) ? 1. : 0.; });

  mfem::Array<int> bdr_attr_list = serac::makeBdrAttributeList(*pmesh, fixed);
  for (int be = 0; be < pmesh->GetNBE(); be++) {
    pmesh->GetBdrElement(be)->SetAttribute(bdr_attr_list[be]);
  }
  pmesh->SetAttributes();

  Array<int> ess_tdof_list;
  Array<int> bdr_attr_is_ess(pmesh->bdr_attributes.Max());
  bdr_attr_is_ess = 0;
  if (bdr_attr_is_ess.Size() > 1) bdr_attr_is_ess[1] = 1;
  pfes_v->GetEssentialTrueDofs(bdr_attr_is_ess, ess_tdof_list);

  // Homogeneous dirchlet bc
  Vector u_ess(ess_tdof_list.Size());
  u_ess = 0.;

  second_order->SetBoundaryConditions(ess_tdof_list, u_ess);

  // Add self-weight
  Vector grav(dim);
  grav    = 0.;
  grav[1] = g;
  VectorConstantCoefficient grav_v_coef(grav);
  // Here rho is the actually rho = rho_const * vf
  ScalarVectorProductCoefficient gravity_load_coef(rho, grav_v_coef);
  auto                           gravity2 = std::make_unique<mfem::VectorDomainLFIntegrator>(gravity_load_coef);
  second_order->AddLinearDomainIntegrator(std::move(gravity2));

  // Make it non rigid body
  ConstantCoefficient lambda(lambda_c);
  ConstantCoefficient mu(mu_c);
  auto                elasticity2 = std::make_unique<mfem::ElasticityIntegrator>(lambda, mu);
  second_order->AddBilinearDomainIntegrator(std::move(elasticity2));

  cout << "output volume fractions :" << endl;

  ParGridFunction gravity_eval(pfes_v.get());
  gravity_eval.ProjectCoefficient(gravity_load_coef);
  gravity_eval.Print();

  double t = 0.;

  std::cout << "t = " << t << std::endl;

  // Store results
  Vector u_prev(x0.GetBlock(0));
  Vector v_prev(x0.GetBlock(1));
  Vector a_prev(x0.GetBlock(2));

  Vector u_next(x0.GetBlock(0));
  Vector v_next(x0.GetBlock(1));
  Vector a_next(x0.GetBlock(2));

  ParGridFunction u_visit(pfes_v.get());
  u_visit = u_next;
  ParGridFunction v_visit(pfes_v.get());
  v_visit = v_next;
  ParGridFunction a_visit(pfes_v.get());
  a_visit = a_next;

  VisItDataCollection visit("FirstOrder", pmesh.get());
  visit.RegisterField("u_next", &u_visit);
  visit.RegisterField("v_next", &v_visit);
  visit.SetCycle(0);
  visit.SetTime(0.);
  visit.Save();

  for (int i = 0; i < num_time_steps; i++) {
    mfem::Vector x0_temp(x0.GetData(), offsets[2]);
    be.Step(x0_temp, t, dt);
    u_visit = x0.GetBlock(0);
    visit.SetTime(t);
    visit.SetCycle(i + 1);
    visit.Save();
  }
}
