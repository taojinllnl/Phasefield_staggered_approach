/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2006 - 2020 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Tao Jin
 *         University of Ottawa, Ottawa, Ontario, Canada
 *         Oct. 2024
 */
//
// This method is created as a reference to compare with the monolithic schemes
// regarding the convergence behavior (how fast or slow each time step converges)
// 1. Staggered solve (alternate minimization)
// 2. Using TBB for stiffness assembly
// 3. Using the history variable approach (maximum positive strain energy) to enforce the
//    irreversibility.
// 4. For the convergence criteria, we use the residual-based criteria. That is,
//    after each staggered iteration, we need to check the residuals for the displacement
//    and the phasefield. The solution converges until both residuals are smaller than
//    the prescribed tolerance.
// 5. Add the adaptively mesh refinement option (Feb. 14th, 2026)

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/base/synchronous_iterator.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgp_monomial.h>
#include <deal.II/fe/mapping_q_eulerian.h>

#include <deal.II/base/timer.h>
#include <deal.II/base/quadrature_point_data.h>
#include <deal.II/base/parameter_handler.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>


#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/packaged_operation.h>
#include <deal.II/lac/precondition_selector.h>
#include <deal.II/lac/solver_selector.h>
#include <deal.II/lac/sparse_direct.h>

#include <deal.II/numerics/error_estimator.h>

#include <deal.II/physics/elasticity/standard_tensors.h>

#include <deal.II/base/quadrature_point_data.h>

#include <deal.II/grid/grid_tools.h>

#include <deal.II/base/work_stream.h>

#include <deal.II/numerics/solution_transfer.h>

#include <fstream>
#include <iostream>
#include <deal.II/base/logstream.h>

#include "SpectrumDecomposition.h"
#include "Utilities.h"

namespace PhaseField
{
  using namespace dealii;

  // body force
  template <int dim>
  void right_hand_side(const std::vector<Point<dim>> &points,
		       std::vector<Tensor<1, dim>> &  values,
		       const double fx,
		       const double fy,
		       const double fz)
  {
    Assert(values.size() == points.size(),
           ExcDimensionMismatch(values.size(), points.size()));
    Assert(dim >= 2, ExcNotImplemented());

    for (unsigned int point_n = 0; point_n < points.size(); ++point_n)
      {
	if (dim == 2)
	  {
	    values[point_n][0] = fx;
	    values[point_n][1] = fy;
	  }
	else
	  {
	    values[point_n][0] = fx;
	    values[point_n][1] = fy;
	    values[point_n][2] = fz;
	  }
      }
  }

  double degradation_function(const double d)
  {
    return (1.0 - d) * (1.0 - d);
  }

  // In the staggered approach, the phase-field subproblem is a linear problem.
  // Therefore, the following function is not used. If we choose a different
  // phase-field degradation function, we have to modify the linear system of
  // the phase-field subproblem manually.
  double degradation_function_derivative(const double d)
  {
    return 2.0 * (d - 1.0);
  }

  // In the staggered approach, since the phase-field subproblem is linear, the
  // 2nd-order derivative of the degradation function is never used.
  double degradation_function_2nd_order_derivative(const double d)
  {
    (void) d;
    return 2.0;
  }

  namespace Parameters
  {
    struct Scenario
    {
      unsigned int m_scenario;
      std::string m_logfile_name;
      bool m_output_iteration_history;
      std::string m_am_convergence_criterion;
      std::string m_type_linear_solver;
      std::string m_refinement_strategy;
      unsigned int m_global_refine_times;
      unsigned int m_local_prerefine_times;
      unsigned int m_max_adaptive_refine_times;
      int m_max_allowed_refinement_level;
      double m_phasefield_refine_threshold;
      double m_allowed_max_h_l_ratio;
      unsigned int m_total_material_regions;
      std::string m_material_file_name;
      int m_reaction_force_face_id;

      static void declare_parameters(ParameterHandler &prm);
      void parse_parameters(ParameterHandler &prm);
    };

    void Scenario::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Scenario");
      {
        prm.declare_entry("Scenario number",
                          "1",
                          Patterns::Integer(0),
                          "Geometry, loading and boundary conditions scenario");

        prm.declare_entry("Log file name",
			  "Output.log",
                          Patterns::FileName(Patterns::FileName::input),
			  "Name of the log file");

        prm.declare_entry("Output iteration history",
			  "yes",
                          Patterns::Selection("yes|no"),
			  "Shall we write iteration history to the log file?");

        prm.declare_entry("AM convergence strategy",
                          "Residual",
                          Patterns::Selection("Residual|Energy|SinglePass"),
                          "Type of convergence strategy for alternate minimization");

        prm.declare_entry("Linear solver type",
                          "Direct",
                          Patterns::Selection("Direct|CG"),
                          "Type of solver used to solve the linear system of the two subproblems");

        prm.declare_entry("Mesh refinement strategy",
                          "adaptive-refine",
                          Patterns::Selection("pre-refine|adaptive-refine"),
                          "Mesh refinement strategy: pre-refine or adaptive-refine");

        prm.declare_entry("Global refinement times",
                          "0",
                          Patterns::Integer(0),
                          "Global refinement times (across the entire domain)");

        prm.declare_entry("Local prerefinement times",
                          "0",
                          Patterns::Integer(0),
			  "Local pre-refinement times (assume crack path is known a priori), "
			  "only refine along the crack path");

        prm.declare_entry("Max adaptive refinement times",
                          "100",
                          Patterns::Integer(0),
                          "Maximum number of adaptive refinement times allowed in each step");

        prm.declare_entry("Max allowed refinement level",
                          "100",
                          Patterns::Integer(0),
                          "Maximum allowed cell refinement level");

        prm.declare_entry("Phasefield refine threshold",
			  "0.8",
			  Patterns::Double(),
			  "Phasefield-based refinement threshold value");

        prm.declare_entry("Allowed max hl ratio",
			  "0.25",
			  Patterns::Double(),
			  "Allowed maximum ratio between mesh size h and length scale l");

        prm.declare_entry("Material regions",
                          "1",
                          Patterns::Integer(0),
                          "Number of material regions");

        prm.declare_entry("Material data file",
                          "1",
                          Patterns::FileName(Patterns::FileName::input),
                          "Material data file");

        prm.declare_entry("Reaction force face ID",
                          "1",
                          Patterns::Integer(),
                          "Face id where reaction forces should be calculated "
                          "(negative integer means not to calculate reaction force)");
      }
      prm.leave_subsection();
    }

    void Scenario::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Scenario");
      {
        m_scenario = prm.get_integer("Scenario number");
        m_logfile_name = prm.get("Log file name");
        m_output_iteration_history = prm.get_bool("Output iteration history");
        m_am_convergence_criterion = prm.get("AM convergence strategy");
        m_type_linear_solver = prm.get("Linear solver type");
        m_refinement_strategy = prm.get("Mesh refinement strategy");
        m_global_refine_times = prm.get_integer("Global refinement times");
        m_local_prerefine_times = prm.get_integer("Local prerefinement times");
        m_max_adaptive_refine_times = prm.get_integer("Max adaptive refinement times");
        m_max_allowed_refinement_level = prm.get_integer("Max allowed refinement level");
        m_phasefield_refine_threshold = prm.get_double("Phasefield refine threshold");
        m_allowed_max_h_l_ratio = prm.get_double("Allowed max hl ratio");
        m_total_material_regions = prm.get_integer("Material regions");
        m_material_file_name = prm.get("Material data file");
        m_reaction_force_face_id = prm.get_integer("Reaction force face ID");
      }
      prm.leave_subsection();
    }

    struct FESystem
    {
      unsigned int m_poly_degree;
      unsigned int m_quad_order;

      static void declare_parameters(ParameterHandler &prm);

      void parse_parameters(ParameterHandler &prm);
    };


    void FESystem::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Finite element system");
      {
        prm.declare_entry("Polynomial degree",
                          "1",
                          Patterns::Integer(0),
                          "Phase field polynomial order");

        prm.declare_entry("Quadrature order",
                          "2",
                          Patterns::Integer(0),
                          "Gauss quadrature order");
      }
      prm.leave_subsection();
    }

    void FESystem::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Finite element system");
      {
        m_poly_degree = prm.get_integer("Polynomial degree");
        m_quad_order  = prm.get_integer("Quadrature order");
      }
      prm.leave_subsection();
    }

    // body force (N/m^3)
    struct BodyForce
    {
      double m_x_component;
      double m_y_component;
      double m_z_component;

      static void declare_parameters(ParameterHandler &prm);

      void parse_parameters(ParameterHandler &prm);
    };

    void BodyForce::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Body force");
      {
        prm.declare_entry("Body force x component",
			  "0.0",
			  Patterns::Double(),
			  "Body force x-component (N/m^3)");

        prm.declare_entry("Body force y component",
			  "0.0",
			  Patterns::Double(),
			  "Body force y-component (N/m^3)");

        prm.declare_entry("Body force z component",
			  "0.0",
			  Patterns::Double(),
			  "Body force z-component (N/m^3)");
      }
      prm.leave_subsection();
    }

    void BodyForce::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Body force");
      {
        m_x_component = prm.get_double("Body force x component");
        m_y_component = prm.get_double("Body force y component");
        m_z_component = prm.get_double("Body force z component");
      }
      prm.leave_subsection();
    }

    struct NonlinearSolver
    {
      unsigned int m_max_am_iteration;
      unsigned int m_max_iterations_newton;

      double       m_tol_u_residual;
      double       m_tol_d_residual;
      double       m_tol_u_incr;
      double       m_tol_d_incr;

      static void declare_parameters(ParameterHandler &prm);

      void parse_parameters(ParameterHandler &prm);
    };

    void NonlinearSolver::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Nonlinear solver");
      {
        prm.declare_entry("Max AM iteration",
                          "100",
                          Patterns::Integer(0),
                          "Maximum allowed number of alternate minimization iterations");

        prm.declare_entry("Phasefield L2 tolerance",
                          "1.0e-3",
                          Patterns::Double(0.0),
                          "Tolerance of L2 norm of phasefield between two iterations "
                          "(only relevant) for energy-based convergence strategy");

        prm.declare_entry("Max iterations Newton-Raphson",
                          "10",
                          Patterns::Integer(0),
                          "Number of Newton-Raphson iterations allowed for "
                          "displacement subproblem");

        prm.declare_entry("Tolerance displacement residual",
                          "1.0e-9",
                          Patterns::Double(0.0),
                          "Displacement residual tolerance");

        prm.declare_entry("Tolerance phasefield residual",
                          "1.0e-9",
                          Patterns::Double(0.0),
                          "Phasefield residual tolerance");

        prm.declare_entry("Tolerance displacement increment",
                          "1.0e-9",
                          Patterns::Double(0.0),
                          "Displacement increment tolerance");

        prm.declare_entry("Tolerance phasefield increment",
                          "1.0e-9",
                          Patterns::Double(0.0),
                          "Phasefield increment tolerance");
      }
      prm.leave_subsection();
    }

    void NonlinearSolver::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Nonlinear solver");
      {
        m_max_am_iteration = prm.get_integer("Max AM iteration");
        m_max_iterations_newton = prm.get_integer("Max iterations Newton-Raphson");

        m_tol_u_residual = prm.get_double("Tolerance displacement residual");
        m_tol_d_residual = prm.get_double("Tolerance phasefield residual");
        m_tol_u_incr     = prm.get_double("Tolerance displacement increment");
        m_tol_d_incr     = prm.get_double("Tolerance phasefield increment");
      }
      prm.leave_subsection();
    }

    struct TimeInfo
    {
      double m_end_time;
      std::string m_time_file_name;

      static void declare_parameters(ParameterHandler &prm);

      void parse_parameters(ParameterHandler &prm);
    };

    void TimeInfo::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Time");
      {
        prm.declare_entry("End time", "1", Patterns::Double(), "End time");

        prm.declare_entry("Time data file",
                          "1",
                          Patterns::FileName(Patterns::FileName::input),
                          "Time data file");
      }
      prm.leave_subsection();
    }

    void TimeInfo::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Time");
      {
        m_end_time = prm.get_double("End time");
        m_time_file_name = prm.get("Time data file");
      }
      prm.leave_subsection();
    }

    struct AllParameters : public Scenario,
	                   public FESystem,
	                   public BodyForce,
			   public NonlinearSolver,
			   public TimeInfo
    {
      AllParameters(const std::string &input_file);

      static void declare_parameters(ParameterHandler &prm);

      void parse_parameters(ParameterHandler &prm);
    };

    AllParameters::AllParameters(const std::string &input_file)
    {
      ParameterHandler prm;
      declare_parameters(prm);
      prm.parse_input(input_file);
      parse_parameters(prm);
    }

    void AllParameters::declare_parameters(ParameterHandler &prm)
    {
      Scenario::declare_parameters(prm);
      FESystem::declare_parameters(prm);
      BodyForce::declare_parameters(prm);
      NonlinearSolver::declare_parameters(prm);
      TimeInfo::declare_parameters(prm);
    }

    void AllParameters::parse_parameters(ParameterHandler &prm)
    {
      Scenario::parse_parameters(prm);
      FESystem::parse_parameters(prm);
      BodyForce::parse_parameters(prm);
      NonlinearSolver::parse_parameters(prm);
      TimeInfo::parse_parameters(prm);
    }
  } // namespace Parameters

  class Time
  {
  public:
    Time(const double time_end)
      : m_timestep(0)
      , m_time_current(0.0)
      , m_time_end(time_end)
      , m_delta_t(0.0)
      , m_magnitude(1.0)
    {}

    virtual ~Time() = default;

    double current() const
    {
      return m_time_current;
    }
    double end() const
    {
      return m_time_end;
    }
    double get_delta_t() const
    {
      return m_delta_t;
    }
    double get_magnitude() const
    {
      return m_magnitude;
    }
    unsigned int get_timestep() const
    {
      return m_timestep;
    }
    void increment(std::vector<std::array<double, 4>> time_table)
    {
      double t_1, t_delta, t_magnitude;
      for (auto & time_group : time_table)
        {
	  t_1 = time_group[1];
	  t_delta = time_group[2];
	  t_magnitude = time_group[3];

	  if (m_time_current < t_1 - 1.0e-6*t_delta)
	    {
	      m_delta_t = t_delta;
	      m_magnitude = t_magnitude;
	      break;
	    }
        }

      m_time_current += m_delta_t;
      ++m_timestep;
    }

  private:
    unsigned int m_timestep;
    double       m_time_current;
    const double m_time_end;
    double m_delta_t;
    double m_magnitude;
  };

  template <int dim>
  class LinearIsotropicElasticityAdditiveSplit
  {
  public:
    LinearIsotropicElasticityAdditiveSplit(const double lame_lambda,
			                   const double lame_mu,
				           const double residual_k,
					   const double length_scale,
					   const double viscosity,
					   const double gc)
      : m_lame_lambda(lame_lambda)
      , m_lame_mu(lame_mu)
      , m_residual_k(residual_k)
      , m_length_scale(length_scale)
      , m_eta(viscosity)
      , m_gc(gc)
      , m_phase_field_value(0.0)
      , m_grad_phasefield(Tensor<1, dim>())
      , m_strain(SymmetricTensor<2, dim>())
      , m_stress(SymmetricTensor<2, dim>())
      , m_mechanical_C(SymmetricTensor<4, dim>())
      , m_strain_energy_positive(0.0)
      , m_strain_energy_negative(0.0)
      , m_strain_energy_total(0.0)
      , m_crack_energy_dissipation(0.0)
    {
      Assert(  ( lame_lambda / (2*(lame_lambda + lame_mu)) <= 0.5)
	     & ( lame_lambda / (2*(lame_lambda + lame_mu)) >=-1.0),
	     ExcInternalError() );
    }

    const SymmetricTensor<4, dim> & get_mechanical_C() const
    {
      return m_mechanical_C;
    }

    const SymmetricTensor<2, dim> & get_cauchy_stress() const
    {
      return m_stress;
    }

    double get_positive_strain_energy() const
    {
      return m_strain_energy_positive;
    }

    double get_negative_strain_energy() const
    {
      return m_strain_energy_negative;
    }

    double get_total_strain_energy() const
    {
      return m_strain_energy_total;
    }

    double get_crack_energy_dissipation() const
    {
      return m_crack_energy_dissipation;
    }

    double get_phase_field_value() const
    {
      return m_phase_field_value;
    }

    const Tensor<1, dim> get_phase_field_gradient() const
    {
      return m_grad_phasefield;
    }

    void update_material_data(const SymmetricTensor<2, dim> & strain,
			      const double phase_field_value,
			      const Tensor<1, dim> & grad_phasefield,
			      const double phase_field_value_previous_step,
			      const double delta_time)
    {
      m_strain = strain;
      m_phase_field_value = phase_field_value;
      m_grad_phasefield = grad_phasefield;
      Vector<double>              eigenvalues(dim);
      std::vector<Tensor<1, dim>> eigenvectors(dim);
      usr_spectrum_decomposition::spectrum_decomposition<dim>(m_strain,
    							      eigenvalues,
    							      eigenvectors);

      SymmetricTensor<2, dim> strain_positive, strain_negative;
      strain_positive = usr_spectrum_decomposition::positive_tensor(eigenvalues, eigenvectors);
      strain_negative = usr_spectrum_decomposition::negative_tensor(eigenvalues, eigenvectors);

      SymmetricTensor<4, dim> projector_positive, projector_negative;
      usr_spectrum_decomposition::positive_negative_projectors(eigenvalues,
    							       eigenvectors,
							       projector_positive,
							       projector_negative);

      SymmetricTensor<2, dim> stress_positive, stress_negative;
      const double degradation = degradation_function(m_phase_field_value) + m_residual_k;
      const double I_1 = trace(m_strain);
      stress_positive = m_lame_lambda * usr_spectrum_decomposition::positive_ramp_function(I_1)
                                      * Physics::Elasticity::StandardTensors<dim>::I
                      + 2 * m_lame_mu * strain_positive;
      stress_negative = m_lame_lambda * usr_spectrum_decomposition::negative_ramp_function(I_1)
                                      * Physics::Elasticity::StandardTensors<dim>::I
      		      + 2 * m_lame_mu * strain_negative;

      m_stress = degradation * stress_positive + stress_negative;

      SymmetricTensor<4, dim> C_positive, C_negative;
      C_positive = m_lame_lambda * usr_spectrum_decomposition::heaviside_function(I_1)
                                 * Physics::Elasticity::StandardTensors<dim>::IxI
		 + 2 * m_lame_mu * projector_positive;
      C_negative = m_lame_lambda * usr_spectrum_decomposition::heaviside_function(-I_1)
                                 * Physics::Elasticity::StandardTensors<dim>::IxI
      		 + 2 * m_lame_mu * projector_negative;
      m_mechanical_C = degradation * C_positive + C_negative;

      m_strain_energy_positive = 0.5 * m_lame_lambda * usr_spectrum_decomposition::positive_ramp_function(I_1)
                                                     * usr_spectrum_decomposition::positive_ramp_function(I_1)
                               + m_lame_mu * strain_positive * strain_positive;

      m_strain_energy_negative = 0.5 * m_lame_lambda * usr_spectrum_decomposition::negative_ramp_function(I_1)
                                                     * usr_spectrum_decomposition::negative_ramp_function(I_1)
                               + m_lame_mu * strain_negative * strain_negative;

      m_strain_energy_total = degradation * m_strain_energy_positive + m_strain_energy_negative;

      m_crack_energy_dissipation = m_gc * (  0.5 / m_length_scale * m_phase_field_value * m_phase_field_value
	                                   + 0.5 * m_length_scale * m_grad_phasefield * m_grad_phasefield)
					 // the term due to viscosity regularization
					 + (m_phase_field_value - phase_field_value_previous_step)
					 * (m_phase_field_value - phase_field_value_previous_step)
					 * 0.5 * m_eta / delta_time;
    }

  private:
    const double m_lame_lambda;
    const double m_lame_mu;
    const double m_residual_k;
    const double m_length_scale;
    const double m_eta;
    const double m_gc;
    double m_phase_field_value;
    Tensor<1, dim> m_grad_phasefield;
    SymmetricTensor<2, dim> m_strain;
    SymmetricTensor<2, dim> m_stress;
    SymmetricTensor<4, dim> m_mechanical_C;
    double m_strain_energy_positive;
    double m_strain_energy_negative;
    double m_strain_energy_total;
    double m_crack_energy_dissipation;
  };


  template <int dim>
  class PointHistory
  {
  public:
    PointHistory()
      : m_length_scale(0.0)
      , m_gc(0.0)
      , m_viscosity(0.0)
      , m_history_max_positive_strain_energy(0.0)
    {}

    virtual ~PointHistory() = default;

    void setup_lqp(const double lame_lambda,
		   const double lame_mu,
		   const double length_scale,
		   const double gc,
		   const double viscosity,
		   const double residual_k)
    {
      m_material =
              std::make_shared<LinearIsotropicElasticityAdditiveSplit<dim>>(lame_lambda,
        	                                                            lame_mu,
								            residual_k,
									    length_scale,
									    viscosity,
									    gc);
      m_history_max_positive_strain_energy = 0.0;
      m_length_scale = length_scale;
      m_gc = gc;
      m_viscosity = viscosity;

      update_field_values(SymmetricTensor<2, dim>(), 0.0, Tensor<1, dim>(), 0.0, 1.0);
    }

    void update_field_values(const SymmetricTensor<2, dim> & strain,
		             const double phase_field_value,
			     const Tensor<1, dim> & grad_phasefield,
			     const double phase_field_value_previous_step,
			     const double delta_time)
    {
      m_material->update_material_data(strain, phase_field_value, grad_phasefield,
				       phase_field_value_previous_step, delta_time);
    }

    void update_history_variable()
    {
      double current_positive_strain_energy = m_material->get_positive_strain_energy();
      m_history_max_positive_strain_energy = std::fmax(m_history_max_positive_strain_energy,
					               current_positive_strain_energy);
    }

    // This is the function used to assign the history variable after remeshing
    void assign_history_variable(double history_variable_value)
    {
      m_history_max_positive_strain_energy = history_variable_value;
    }

    double get_current_positive_strain_energy() const
    {
      return m_material->get_positive_strain_energy();
    }

    const SymmetricTensor<4, dim> & get_mechanical_C() const
    {
      return m_material->get_mechanical_C();
    }

    const SymmetricTensor<2, dim> & get_cauchy_stress() const
    {
      return m_material->get_cauchy_stress();
    }

    double get_total_strain_energy() const
    {
      return m_material->get_total_strain_energy();
    }

    double get_crack_energy_dissipation() const
    {
      return m_material->get_crack_energy_dissipation();
    }

    double get_phase_field_value() const
    {
      return m_material->get_phase_field_value();
    }

    const Tensor<1, dim> get_phase_field_gradient() const
    {
      return m_material->get_phase_field_gradient();
    }

    double get_history_max_positive_strain_energy() const
    {
      return m_history_max_positive_strain_energy;
    }

    double get_length_scale() const
    {
      return m_length_scale;
    }

    double get_critical_energy_release_rate() const
    {
      return m_gc;
    }

    double get_viscosity() const
    {
      return m_viscosity;
    }
  private:
    std::shared_ptr<LinearIsotropicElasticityAdditiveSplit<dim>> m_material;
    double m_length_scale;
    double m_gc;
    double m_viscosity;
    double m_history_max_positive_strain_energy;
  };

  template <int dim>
  class PhaseFieldSplitSolve
  {
  public:
    PhaseFieldSplitSolve(const std::string &input_file);

    virtual ~PhaseFieldSplitSolve() = default;
    void run();

  private:
    using IteratorTuple =
      std::tuple<typename DoFHandler<dim>::active_cell_iterator,
                 typename DoFHandler<dim>::active_cell_iterator>;

    using IteratorPair = SynchronousIterators<IteratorTuple>;

    struct PerTaskData_ASM_phasefield;
    struct ScratchData_ASM_phasefield;

    struct PerTaskData_ASM_displacement;
    struct ScratchData_ASM_displacement;

    struct PerTaskData_RHS_phasefield;
    struct ScratchData_RHS_phasefield;

    struct PerTaskData_RHS_displacement;
    struct ScratchData_RHS_displacement;

    struct PerTaskData_UQPH;
    struct ScratchData_UQPH;

    Parameters::AllParameters m_parameters;
    Triangulation<dim> m_triangulation;

    CellDataStorage<typename Triangulation<dim>::cell_iterator,
                    PointHistory<dim>>
      m_quadrature_point_history;

    Time                m_time;
    std::ofstream m_logfile;
    mutable TimerOutput m_timer;

    DoFHandler<dim> m_dof_handler_phasefield;
    DoFHandler<dim> m_dof_handler_displacement;

    FESystem<dim> m_fe_phasefield;
    FESystem<dim> m_fe_displacement;

    const QGauss<dim>     m_qf_cell;
    const QGauss<dim - 1> m_qf_face;
    const unsigned int    m_n_q_points;


    AffineConstraints<double> m_constraints_phasefield;
    AffineConstraints<double> m_constraints_displacement;

    SparsityPattern m_sparsity_pattern_phasefield;
    SparsityPattern m_sparsity_pattern_displacement;

    SparseMatrix<double> m_system_matrix_phasefield;
    SparseMatrix<double> m_system_matrix_displacement;

    Vector<double> m_system_rhs_phasefield;
    Vector<double> m_system_rhs_displacement;

    Vector<double> m_solution_phasefield;
    Vector<double> m_solution_displacement;
    Vector<double> m_solution_previous_timestep_phasefield;
    Vector<double> m_solution_previous_timestep_displacement;

    double m_vol_reference;

    std::map<unsigned int, std::vector<double>> m_material_data;

    std::vector<std::pair<double, std::vector<double>>> m_history_reaction_force;
    std::vector<std::pair<double, std::array<double, 3>>> m_history_energy;

    struct Errors
    {
      Errors()
        : m_norm(1.0)
      {}

      void reset()
      {
        m_norm = 1.0;
      }
      void normalize(const Errors &rhs)
      {
        if (rhs.m_norm != 0.0)
          m_norm /= rhs.m_norm;
      }

      double m_norm;
    };

    Errors m_error_residual_displacement, m_error_update_displacement;

    void get_error_residual_displacement(Errors &error_residual);
    void get_error_update_displacement(const Vector<double> &newton_update,
                                       Errors & error_update);

    void make_grid();
    void make_grid_case_1();
    void make_grid_case_2();
    void make_grid_case_3();
    void make_grid_case_4();
    void make_grid_case_9();

    void setup_system();
    void setup_system_phasefield();
    void setup_system_displacement();

    void make_constraints_phasefield();
    void make_constraints_displacement(const unsigned int it_nr,
				       const unsigned int itr_stagger);

    void assemble_system_phasefield();
    void assemble_system_displacement();

    void assemble_rhs_phasefield();
    void assemble_rhs_displacement();

    void phasefield_step();
    void solve_linear_system_phasefield();
    unsigned int displacement_step(unsigned int itr_stagger);
    unsigned int solve_nonlinear_displacement_newton_raphson(Vector<double> &solution_delta_displacement,
				    		             unsigned int itr_stagger);
    void solve_linear_system_displacement(Vector<double> & newton_update);
    void update_history_field_step();
    void update_phasefield();

    void output_results() const;
    void assemble_system_one_cell_phasefield(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                             ScratchData_ASM_phasefield & scratch,
                                             PerTaskData_ASM_phasefield & data) const;
    void assemble_rhs_one_cell_phasefield(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                          ScratchData_RHS_phasefield & scratch,
                                          PerTaskData_RHS_phasefield & data) const;
    void assemble_system_one_cell_displacement(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                               ScratchData_ASM_displacement & scratch,
                                               PerTaskData_ASM_displacement & data) const;
    void assemble_rhs_one_cell_displacement(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                            ScratchData_RHS_displacement & scratch,
                                            PerTaskData_RHS_displacement & data) const;


    void setup_qph();

    void update_qph_incremental(const Vector<double> &solution_delta_displacement);

    void update_qph_incremental_one_cell(
      const IteratorPair & synchronous_iterators,
      ScratchData_UQPH   & scratch,
      PerTaskData_UQPH   & data);

    void copy_local_to_global_UQPH(const PerTaskData_UQPH & /*data*/)
    {}

    bool local_refine_and_solution_transfer();

    Vector<double>
    get_total_solution_u(const Vector<double> &solution_delta_displacement) const;

    // Should not make this function const
    void read_material_data(const std::string &data_file,
			    const unsigned int total_material_regions);

    void read_time_data(const std::string &data_file,
    		        std::vector<std::array<double, 4>> & time_table);

    void print_conv_header();

    void print_parameter_information();

    void calculate_reaction_force(unsigned int face_ID);

    void write_history_data();

    double calculate_energy_functional() const;

    std::pair<double, double>
      calculate_total_strain_energy_and_crack_energy_dissipation() const;
  }; // class PhaseFieldSplitSolve


  template <int dim>
  void PhaseFieldSplitSolve<dim>::get_error_residual_displacement(Errors &error_residual)
  {
    Vector<double> error_res(m_dof_handler_displacement.n_dofs());

    for (unsigned int i = 0; i < m_dof_handler_displacement.n_dofs(); ++i)
      if (!m_constraints_displacement.is_constrained(i))
        error_res(i) = m_system_rhs_displacement(i);

    error_residual.m_norm = error_res.l2_norm();
  }

  template <int dim>
  void PhaseFieldSplitSolve<dim>::get_error_update_displacement(const Vector<double> &newton_update,
                                                                Errors & error_update)
  {
    Vector<double> error_ud(m_dof_handler_displacement.n_dofs());
    for (unsigned int i = 0; i < m_dof_handler_displacement.n_dofs(); ++i)
      if (!m_constraints_displacement.is_constrained(i))
        error_ud(i) = newton_update(i);

    error_update.m_norm = error_ud.l2_norm();
  }

  template <int dim>
  void PhaseFieldSplitSolve<dim>::read_material_data(const std::string &data_file,
				                     const unsigned int total_material_regions)
  {
    std::ifstream myfile (data_file);

    double lame_lambda, lame_mu, length_scale, gc, viscosity, residual_k;
    int material_region;
    double poisson_ratio;
    if (myfile.is_open())
      {
	m_logfile << "Reading material data file ..." << std::endl;

        while ( myfile >> material_region
                       >> lame_lambda
		       >> lame_mu
		       >> length_scale
		       >> gc
		       >> viscosity
		       >> residual_k)
          {
            m_material_data[material_region] = {lame_lambda,
        	                                lame_mu,
						length_scale,
						gc,
						viscosity,
                                                residual_k};
            poisson_ratio = lame_lambda / (2*(lame_lambda + lame_mu));
            Assert( (poisson_ratio <= 0.5)&(poisson_ratio >=-1.0) , ExcInternalError());

            m_logfile << "\tRegion " << material_region << " : " << std::endl;
            m_logfile << "\t\tLame lambda = " << lame_lambda << std::endl;
            m_logfile << "\t\tLame mu = "  << lame_mu << std::endl;
            m_logfile << "\t\tPoisson ratio = "  << poisson_ratio << std::endl;
            m_logfile << "\t\tPhase field length scale (l) = " << length_scale << std::endl;
            m_logfile << "\t\tCritical energy release rate (gc) = "  << gc << std::endl;
            m_logfile << "\t\tViscosity for regularization (eta) = "  << viscosity << std::endl;
            m_logfile << "\t\tResidual_k (k) = "  << residual_k << std::endl;
          }

        if (m_material_data.size() != total_material_regions)
          {
            m_logfile << "Material data file has " << m_material_data.size() << " rows. However, "
        	      << "the mesh has " << total_material_regions << " material regions."
		      << std::endl;
            Assert(m_material_data.size() == total_material_regions,
                       ExcDimensionMismatch(m_material_data.size(), total_material_regions));
          }
        myfile.close();
      }
    else
      {
	m_logfile << "Material data file : " << data_file << " not exist!" << std::endl;
	Assert(false, ExcMessage("Failed to read material data file"));
      }
  }

  template <int dim>
  void PhaseFieldSplitSolve<dim>::read_time_data(const std::string &data_file,
				                 std::vector<std::array<double, 4>> & time_table)
  {
    std::ifstream myfile (data_file);

    double t_0, t_1, delta_t, t_magnitude;

    if (myfile.is_open())
      {
	m_logfile << "Reading time data file ..." << std::endl;

	while ( myfile >> t_0
		       >> t_1
		       >> delta_t
		       >> t_magnitude)
	  {
	    Assert( t_0 < t_1,
		    ExcMessage("For each time pair, "
			       "the start time should be smaller than the end time"));
	    time_table.push_back({{t_0, t_1, delta_t, t_magnitude}});
	  }

	Assert(std::fabs(t_1 - m_parameters.m_end_time) < 1.0e-9,
	       ExcMessage("End time in time table is inconsistent with input data in parameters.prm"));

	Assert(time_table.size() > 0,
	       ExcMessage("Time data file is empty."));
	myfile.close();
      }
    else
      {
        m_logfile << "Time data file : " << data_file << " not exist!" << std::endl;
        Assert(false, ExcMessage("Failed to read time data file"));
      }

    for (auto & time_group : time_table)
      {
	m_logfile << "\t\t"
	          << time_group[0] << ",\t"
	          << time_group[1] << ",\t"
		  << time_group[2] << ",\t"
		  << time_group[3] << std::endl;
      }
  }

  template <int dim>
  void PhaseFieldSplitSolve<dim>::setup_qph()
  {
    m_logfile << "\t\tSetting up quadrature point data ("
	      << m_n_q_points
	      << " points per cell)" << std::endl;

    m_quadrature_point_history.clear();
    for (auto const & cell : m_triangulation.active_cell_iterators())
      {
	m_quadrature_point_history.initialize(cell, m_n_q_points);
      }

    unsigned int material_id;
    double lame_lambda = 0.0;
    double lame_mu = 0.0;
    double length_scale = 0.0;
    double gc = 0.0;
    double viscosity = 0.0;
    double residual_k = 0.0;

    for (const auto &cell : m_triangulation.active_cell_iterators())
      {
        material_id = cell->material_id();
        if (m_material_data.find(material_id) != m_material_data.end())
          {
            lame_lambda                = m_material_data[material_id][0];
            lame_mu                    = m_material_data[material_id][1];
            length_scale               = m_material_data[material_id][2];
            gc                         = m_material_data[material_id][3];
            viscosity                  = m_material_data[material_id][4];
            residual_k                 = m_material_data[material_id][5];
	  }
        else
          {
            m_logfile << "Could not find material data for material id: " << material_id << std::endl;
            AssertThrow(false, ExcMessage("Could not find material data for material id."));
          }

        const std::vector<std::shared_ptr<PointHistory<dim>>> lqph =
          m_quadrature_point_history.get_data(cell);

        Assert(lqph.size() == m_n_q_points, ExcInternalError());

        for (unsigned int q_point = 0; q_point < m_n_q_points; ++q_point)
          lqph[q_point]->setup_lqp(lame_lambda, lame_mu, length_scale,
				   gc, viscosity, residual_k);
      }
  }

  template <int dim>
  Vector<double> PhaseFieldSplitSolve<dim>::get_total_solution_u(
    const Vector<double> &solution_delta_displacement) const
  {
    Vector<double> solution_total_displacement(m_solution_displacement);
    solution_total_displacement += solution_delta_displacement;
    return solution_total_displacement;
  }

  template <int dim>
  void PhaseFieldSplitSolve<dim>::update_phasefield()
  {
    Vector<double> solution_delta_displacement(m_dof_handler_displacement.n_dofs());
    solution_delta_displacement = 0.0;
    update_qph_incremental(solution_delta_displacement);
  }

  template <int dim>
  void
  PhaseFieldSplitSolve<dim>::update_qph_incremental(const Vector<double> &solution_delta_displacement)
  {
    m_timer.enter_subsection("Update QPH data");

    const Vector<double> solution_total_displacement(get_total_solution_u(solution_delta_displacement));

    const UpdateFlags uf_UQPH(update_values | update_gradients);
    PerTaskData_UQPH  per_task_data_UQPH;
    ScratchData_UQPH  scratch_data_UQPH(m_fe_displacement,
					m_fe_phasefield,
					m_qf_cell,
					uf_UQPH,
					solution_total_displacement,
					m_solution_phasefield,
					m_solution_previous_timestep_phasefield);

    auto worker = [this](const IteratorPair & synchronous_iterators,
	                 ScratchData_UQPH & scratch,
	                 PerTaskData_UQPH & data)
      {
        this->update_qph_incremental_one_cell(synchronous_iterators, scratch, data);
      };

    auto copier = [this](const PerTaskData_UQPH &data)
      {
        this->copy_local_to_global_UQPH(data);
      };

    WorkStream::run(
	IteratorPair(IteratorTuple(m_dof_handler_displacement.begin_active(),
				   m_dof_handler_phasefield.begin_active())),
	IteratorPair(IteratorTuple(m_dof_handler_displacement.end(),
				   m_dof_handler_phasefield.end())),
	worker,
	copier,
	scratch_data_UQPH,
	per_task_data_UQPH);

    m_timer.leave_subsection();
  }

  template <int dim>
  struct PhaseFieldSplitSolve<dim>::PerTaskData_UQPH
  {
    void reset()
    {}
  };


  template <int dim>
  struct PhaseFieldSplitSolve<dim>::ScratchData_UQPH
  {
    const Vector<double> & m_solution_displacement_UQPH;
    const Vector<double> & m_solution_phasefield_UQPH;
    const Vector<double> & m_solution_previous_phasefield_UQPH;

    std::vector<SymmetricTensor<2, dim>> m_solution_symm_grads_u_cell;
    std::vector<double>         m_solution_values_phasefield_cell;
    std::vector<Tensor<1, dim>> m_solution_grad_phasefield_cell;
    std::vector<double>         m_old_phasefield_cell;

    FEValues<dim> m_fe_values_displacement;
    FEValues<dim> m_fe_values_phasefield;


    ScratchData_UQPH(const FiniteElement<dim> & fe_cell_displacement,
		     const FiniteElement<dim> & fe_cell_phasefield,
                     const QGauss<dim> &        qf_cell,
                     const UpdateFlags          uf_cell,
                     const Vector<double> &solution_total_displacement,
		     const Vector<double> &solution_total_phasefield,
		     const Vector<double> &solution_previous_phasefield)
      : m_solution_displacement_UQPH(solution_total_displacement)
      , m_solution_phasefield_UQPH(solution_total_phasefield)
      , m_solution_previous_phasefield_UQPH(solution_previous_phasefield)
      , m_solution_symm_grads_u_cell(qf_cell.size())
      , m_solution_values_phasefield_cell(qf_cell.size())
      , m_solution_grad_phasefield_cell(qf_cell.size())
      , m_old_phasefield_cell(qf_cell.size())
      , m_fe_values_displacement(fe_cell_displacement, qf_cell, uf_cell)
      , m_fe_values_phasefield(fe_cell_phasefield, qf_cell, uf_cell)
    {}

    ScratchData_UQPH(const ScratchData_UQPH &rhs)
      : m_solution_displacement_UQPH(rhs.m_solution_displacement_UQPH)
      , m_solution_phasefield_UQPH(rhs.m_solution_phasefield_UQPH)
      , m_solution_previous_phasefield_UQPH(rhs.m_solution_previous_phasefield_UQPH)
      , m_solution_symm_grads_u_cell(rhs.m_solution_symm_grads_u_cell)
      , m_solution_values_phasefield_cell(rhs.m_solution_values_phasefield_cell)
      , m_solution_grad_phasefield_cell(rhs.m_solution_grad_phasefield_cell)
      , m_old_phasefield_cell(rhs.m_old_phasefield_cell)
      , m_fe_values_displacement(rhs.m_fe_values_displacement.get_fe(),
                                 rhs.m_fe_values_displacement.get_quadrature(),
                                 rhs.m_fe_values_displacement.get_update_flags())
      , m_fe_values_phasefield(rhs.m_fe_values_phasefield.get_fe(),
                               rhs.m_fe_values_phasefield.get_quadrature(),
                               rhs.m_fe_values_phasefield.get_update_flags())
    {}

    void reset()
    {
      const unsigned int n_q_points = m_solution_symm_grads_u_cell.size();
      for (unsigned int q = 0; q < n_q_points; ++q)
        {
          m_solution_symm_grads_u_cell[q]  = 0.0;
          m_solution_values_phasefield_cell[q] = 0.0;
          m_solution_grad_phasefield_cell[q] = 0.0;
          m_old_phasefield_cell[q] = 0.0;
        }
    }
  };

  template <int dim>
  void PhaseFieldSplitSolve<dim>::update_qph_incremental_one_cell(
    const IteratorPair & synchronous_iterators,
    ScratchData_UQPH & scratch,
    PerTaskData_UQPH & /*data*/)
  {
    scratch.reset();

    scratch.m_fe_values_displacement.reinit(std::get<0>(*synchronous_iterators));
    scratch.m_fe_values_phasefield.reinit(std::get<1>(*synchronous_iterators));

    const std::vector<std::shared_ptr<PointHistory<dim>>> lqph =
      m_quadrature_point_history.get_data(std::get<0>(*synchronous_iterators));
    Assert(lqph.size() == m_n_q_points, ExcInternalError());

    const FEValuesExtractors::Vector displacement(0);

    scratch.m_fe_values_displacement[displacement].get_function_symmetric_gradients(
      scratch.m_solution_displacement_UQPH, scratch.m_solution_symm_grads_u_cell);
    scratch.m_fe_values_phasefield.get_function_values(
      scratch.m_solution_phasefield_UQPH, scratch.m_solution_values_phasefield_cell);
    scratch.m_fe_values_phasefield.get_function_gradients(
      scratch.m_solution_phasefield_UQPH, scratch.m_solution_grad_phasefield_cell);
    scratch.m_fe_values_phasefield.get_function_values(
      scratch.m_solution_previous_phasefield_UQPH, scratch.m_old_phasefield_cell);

    const double delta_time = m_time.get_delta_t();

    for (const unsigned int q_point :
         scratch.m_fe_values_displacement.quadrature_point_indices())
      lqph[q_point]->update_field_values(scratch.m_solution_symm_grads_u_cell[q_point],
                                         scratch.m_solution_values_phasefield_cell[q_point],
					 scratch.m_solution_grad_phasefield_cell[q_point],
					 scratch.m_old_phasefield_cell[q_point],
					 delta_time);
  }

  template <int dim>
  struct PhaseFieldSplitSolve<dim>::PerTaskData_ASM_phasefield
  {
    FullMatrix<double>                   m_cell_matrix;
    Vector<double>                       m_cell_rhs;
    std::vector<types::global_dof_index> m_local_dof_indices;

    PerTaskData_ASM_phasefield(const unsigned int dofs_per_cell)
      : m_cell_matrix(dofs_per_cell, dofs_per_cell)
      , m_cell_rhs(dofs_per_cell)
      , m_local_dof_indices(dofs_per_cell)
    {}

    void reset()
    {
      m_cell_matrix = 0.0;
      m_cell_rhs    = 0.0;
    }
  };

  template <int dim>
  struct PhaseFieldSplitSolve<dim>::PerTaskData_RHS_phasefield
  {
    Vector<double>                       m_cell_rhs;
    std::vector<types::global_dof_index> m_local_dof_indices;

    PerTaskData_RHS_phasefield(const unsigned int dofs_per_cell)
      : m_cell_rhs(dofs_per_cell)
      , m_local_dof_indices(dofs_per_cell)
    {}

    void reset()
    {
      m_cell_rhs    = 0.0;
    }
  };

  template <int dim>
  struct PhaseFieldSplitSolve<dim>::PerTaskData_ASM_displacement
  {
    FullMatrix<double>                   m_cell_matrix;
    Vector<double>                       m_cell_rhs;
    std::vector<types::global_dof_index> m_local_dof_indices;

    PerTaskData_ASM_displacement(const unsigned int dofs_per_cell)
      : m_cell_matrix(dofs_per_cell, dofs_per_cell)
      , m_cell_rhs(dofs_per_cell)
      , m_local_dof_indices(dofs_per_cell)
    {}

    void reset()
    {
      m_cell_matrix = 0.0;
      m_cell_rhs    = 0.0;
    }
  };

  template <int dim>
  struct PhaseFieldSplitSolve<dim>::PerTaskData_RHS_displacement
  {
    Vector<double>                       m_cell_rhs;
    std::vector<types::global_dof_index> m_local_dof_indices;

    PerTaskData_RHS_displacement(const unsigned int dofs_per_cell)
      : m_cell_rhs(dofs_per_cell)
      , m_local_dof_indices(dofs_per_cell)
    {}

    void reset()
    {
      m_cell_rhs    = 0.0;
    }
  };

  template <int dim>
  struct PhaseFieldSplitSolve<dim>::ScratchData_ASM_phasefield
  {
    FEValues<dim>     m_fe_values;

    std::vector<std::vector<double>>                  m_Nx;  // shape function values for phase-field
    std::vector<std::vector<Tensor<1, dim>>>          m_grad_Nx;
    const Vector<double>&       m_solution_previous_phasefield;
    std::vector<double>         m_old_phasefield;

    ScratchData_ASM_phasefield(const FiniteElement<dim> &fe_cell,
                    const QGauss<dim> &       qf_cell,
                    const UpdateFlags         uf_cell,
		    const Vector<double>&     solution_old_phasefield)
      : m_fe_values(fe_cell, qf_cell, uf_cell)
      , m_Nx(qf_cell.size(),
	     std::vector<double>(fe_cell.n_dofs_per_cell()))
      , m_grad_Nx(qf_cell.size(),
		  std::vector<Tensor<1, dim>>(fe_cell.n_dofs_per_cell()))
      , m_solution_previous_phasefield(solution_old_phasefield)
      , m_old_phasefield(qf_cell.size())
    {}

    ScratchData_ASM_phasefield(const ScratchData_ASM_phasefield &rhs)
      : m_fe_values(rhs.m_fe_values.get_fe(),
                    rhs.m_fe_values.get_quadrature(),
                    rhs.m_fe_values.get_update_flags())
      , m_Nx(rhs.m_Nx)
      , m_grad_Nx(rhs.m_grad_Nx)
      , m_solution_previous_phasefield(rhs.m_solution_previous_phasefield)
      , m_old_phasefield(rhs.m_old_phasefield)
    {}

    void reset()
    {
      const unsigned int n_q_points      = m_Nx.size();
      const unsigned int n_dofs_per_cell = m_Nx[0].size();
      for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
        {
          Assert(m_Nx[q_point].size() == n_dofs_per_cell, ExcInternalError());

          Assert(m_grad_Nx[q_point].size() == n_dofs_per_cell,
                 ExcInternalError());

          m_old_phasefield[q_point] = 0.0;
          for (unsigned int k = 0; k < n_dofs_per_cell; ++k)
            {
              m_Nx[q_point][k]           = 0.0;
              m_grad_Nx[q_point][k]      = 0.0;
            }
        }
    }
  };

  template <int dim>
  struct PhaseFieldSplitSolve<dim>::ScratchData_RHS_phasefield
  {
    FEValues<dim>     m_fe_values;

    std::vector<std::vector<double>>                  m_Nx;  // shape function values for phase-field
    std::vector<std::vector<Tensor<1, dim>>>          m_grad_Nx;
    const Vector<double>&       m_solution_previous_phasefield;
    std::vector<double>         m_old_phasefield;

    ScratchData_RHS_phasefield(const FiniteElement<dim> &fe_cell,
                    const QGauss<dim> &       qf_cell,
                    const UpdateFlags         uf_cell,
		    const Vector<double>&     solution_old_phasefield)
      : m_fe_values(fe_cell, qf_cell, uf_cell)
      , m_Nx(qf_cell.size(),
	     std::vector<double>(fe_cell.n_dofs_per_cell()))
      , m_grad_Nx(qf_cell.size(),
		  std::vector<Tensor<1, dim>>(fe_cell.n_dofs_per_cell()))
      , m_solution_previous_phasefield(solution_old_phasefield)
      , m_old_phasefield(qf_cell.size())
    {}

    ScratchData_RHS_phasefield(const ScratchData_RHS_phasefield &rhs)
      : m_fe_values(rhs.m_fe_values.get_fe(),
                    rhs.m_fe_values.get_quadrature(),
                    rhs.m_fe_values.get_update_flags())
      , m_Nx(rhs.m_Nx)
      , m_grad_Nx(rhs.m_grad_Nx)
      , m_solution_previous_phasefield(rhs.m_solution_previous_phasefield)
      , m_old_phasefield(rhs.m_old_phasefield)
    {}

    void reset()
    {
      const unsigned int n_q_points      = m_Nx.size();
      const unsigned int n_dofs_per_cell = m_Nx[0].size();
      for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
        {
          Assert(m_Nx[q_point].size() == n_dofs_per_cell, ExcInternalError());

          Assert(m_grad_Nx[q_point].size() == n_dofs_per_cell,
                 ExcInternalError());

          m_old_phasefield[q_point] = 0.0;
          for (unsigned int k = 0; k < n_dofs_per_cell; ++k)
            {
              m_Nx[q_point][k]           = 0.0;
              m_grad_Nx[q_point][k]      = 0.0;
            }
        }
    }
  };

  template <int dim>
  struct PhaseFieldSplitSolve<dim>::ScratchData_ASM_displacement
  {
    FEValues<dim>     m_fe_values;
    FEFaceValues<dim> m_fe_face_values;

    std::vector<std::vector<Tensor<1, dim>>>          m_Nx;  // shape function values for displacement
    std::vector<std::vector<Tensor<2, dim>>>          m_grad_Nx;
    std::vector<std::vector<SymmetricTensor<2, dim>>> m_symm_grad_Nx;

    ScratchData_ASM_displacement(const FiniteElement<dim> &fe_cell,
                                 const QGauss<dim> &       qf_cell,
                                 const UpdateFlags         uf_cell,
				 const QGauss<dim - 1> &   qf_face,
				 const UpdateFlags         uf_face)
      : m_fe_values(fe_cell, qf_cell, uf_cell)
      , m_fe_face_values(fe_cell, qf_face, uf_face)
      , m_Nx(qf_cell.size(),
	     std::vector<Tensor<1, dim>>(fe_cell.n_dofs_per_cell()))
      , m_grad_Nx(qf_cell.size(),
		  std::vector<Tensor<2, dim>>(fe_cell.n_dofs_per_cell()))
      , m_symm_grad_Nx(qf_cell.size(),
		       std::vector<SymmetricTensor<2, dim>>(fe_cell.n_dofs_per_cell()))
    {}

    ScratchData_ASM_displacement(const ScratchData_ASM_displacement &rhs)
      : m_fe_values(rhs.m_fe_values.get_fe(),
                    rhs.m_fe_values.get_quadrature(),
                    rhs.m_fe_values.get_update_flags())
      , m_fe_face_values(rhs.m_fe_face_values.get_fe(),
                         rhs.m_fe_face_values.get_quadrature(),
                         rhs.m_fe_face_values.get_update_flags())
      , m_Nx(rhs.m_Nx)
      , m_grad_Nx(rhs.m_grad_Nx)
      , m_symm_grad_Nx(rhs.m_symm_grad_Nx)
    {}

    void reset()
    {
      const unsigned int n_q_points      = m_Nx.size();
      const unsigned int n_dofs_per_cell = m_Nx[0].size();
      for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
        {
          Assert(m_Nx[q_point].size() == n_dofs_per_cell, ExcInternalError());

          Assert(m_grad_Nx[q_point].size() == n_dofs_per_cell,
                 ExcInternalError());
          Assert(m_symm_grad_Nx[q_point].size() == n_dofs_per_cell,
                 ExcInternalError());

          for (unsigned int k = 0; k < n_dofs_per_cell; ++k)
            {
              m_Nx[q_point][k]           = 0.0;
              m_grad_Nx[q_point][k]      = 0.0;
              m_symm_grad_Nx[q_point][k] = 0.0;
            }
        }
    }
  };

  template <int dim>
  struct PhaseFieldSplitSolve<dim>::ScratchData_RHS_displacement
  {
    FEValues<dim>     m_fe_values;
    FEFaceValues<dim> m_fe_face_values;

    std::vector<std::vector<Tensor<1, dim>>>          m_Nx;  // shape function values for displacement
    std::vector<std::vector<Tensor<2, dim>>>          m_grad_Nx;
    std::vector<std::vector<SymmetricTensor<2, dim>>> m_symm_grad_Nx;

    ScratchData_RHS_displacement(const FiniteElement<dim> &fe_cell,
                                 const QGauss<dim> &       qf_cell,
                                 const UpdateFlags         uf_cell,
				 const QGauss<dim - 1> &   qf_face,
				 const UpdateFlags         uf_face)
      : m_fe_values(fe_cell, qf_cell, uf_cell)
      , m_fe_face_values(fe_cell, qf_face, uf_face)
      , m_Nx(qf_cell.size(),
	     std::vector<Tensor<1, dim>>(fe_cell.n_dofs_per_cell()))
      , m_grad_Nx(qf_cell.size(),
		  std::vector<Tensor<2, dim>>(fe_cell.n_dofs_per_cell()))
      , m_symm_grad_Nx(qf_cell.size(),
		       std::vector<SymmetricTensor<2, dim>>(fe_cell.n_dofs_per_cell()))
    {}

    ScratchData_RHS_displacement(const ScratchData_RHS_displacement &rhs)
      : m_fe_values(rhs.m_fe_values.get_fe(),
                    rhs.m_fe_values.get_quadrature(),
                    rhs.m_fe_values.get_update_flags())
      , m_fe_face_values(rhs.m_fe_face_values.get_fe(),
                         rhs.m_fe_face_values.get_quadrature(),
                         rhs.m_fe_face_values.get_update_flags())
      , m_Nx(rhs.m_Nx)
      , m_grad_Nx(rhs.m_grad_Nx)
      , m_symm_grad_Nx(rhs.m_symm_grad_Nx)
    {}

    void reset()
    {
      const unsigned int n_q_points      = m_Nx.size();
      const unsigned int n_dofs_per_cell = m_Nx[0].size();
      for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
        {
          Assert(m_Nx[q_point].size() == n_dofs_per_cell, ExcInternalError());

          Assert(m_grad_Nx[q_point].size() == n_dofs_per_cell,
                 ExcInternalError());
          Assert(m_symm_grad_Nx[q_point].size() == n_dofs_per_cell,
                 ExcInternalError());

          for (unsigned int k = 0; k < n_dofs_per_cell; ++k)
            {
              m_Nx[q_point][k]           = 0.0;
              m_grad_Nx[q_point][k]      = 0.0;
              m_symm_grad_Nx[q_point][k] = 0.0;
            }
        }
    }
  };

  // constructor has no return type
  template <int dim>
  PhaseFieldSplitSolve<dim>::PhaseFieldSplitSolve(const std::string &input_file)
    : m_parameters(input_file)
    , m_triangulation(Triangulation<dim>::maximum_smoothing)
    , m_time(m_parameters.m_end_time)
    , m_logfile(m_parameters.m_logfile_name)
    , m_timer(m_logfile, TimerOutput::summary, TimerOutput::wall_times)
    , m_dof_handler_phasefield(m_triangulation)
    , m_dof_handler_displacement(m_triangulation)
    , m_fe_phasefield(FE_Q<dim>(m_parameters.m_poly_degree),
	              1)
    , m_fe_displacement(FE_Q<dim>(m_parameters.m_poly_degree),
	                dim)
    , m_qf_cell(m_parameters.m_quad_order)
    , m_qf_face(m_parameters.m_quad_order)
    , m_n_q_points(m_qf_cell.size())
    , m_vol_reference(0.0)
  {}

  template <int dim>
  void PhaseFieldSplitSolve<dim>::make_grid()
  {
    if (m_parameters.m_scenario == 1)
      make_grid_case_1();
    else if (m_parameters.m_scenario == 2)
      make_grid_case_2();
    else if (m_parameters.m_scenario == 3)
      make_grid_case_3();
    else if (m_parameters.m_scenario == 4)
      make_grid_case_4();
    else if (m_parameters.m_scenario == 9)
      make_grid_case_9();
    else
      Assert(false, ExcMessage("The scenario has not been implemented!"));

    m_logfile << "\t\tTriangulation:"
              << "\n\t\t\tNumber of active cells: "
              << m_triangulation.n_active_cells()
              << "\n\t\t\tNumber of used vertices: "
              << m_triangulation.n_used_vertices()
	      << std::endl;

    std::ofstream out("original_mesh.vtu");
    GridOut       grid_out;
    grid_out.write_vtu(m_triangulation, out);

    m_vol_reference = GridTools::volume(m_triangulation);
    m_logfile << "\t\tGrid:\n\t\t\tReference volume: " << m_vol_reference << std::endl;
  }

  template <int dim>
  void PhaseFieldSplitSolve<dim>::make_grid_case_1()
  {
    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;
    m_logfile << "\t\t\tSquare tension (unstructured)" << std::endl;
    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;

    AssertThrow(dim==2, ExcMessage("The dimension has to be 2D!"));

    GridIn<dim> gridin;
    gridin.attach_triangulation(m_triangulation);
    std::ifstream f("square_tension_unstructured.msh");
    gridin.read_msh(f);

    for (const auto &cell : m_triangulation.active_cell_iterators())
      for (const auto &face : cell->face_iterators())
	{
	  if (face->at_boundary() == true)
	    {
	      if (std::fabs(face->center()[1] + 0.5 ) < 1.0e-9 )
		face->set_boundary_id(0);
	      else if (std::fabs(face->center()[1] - 0.5 ) < 1.0e-9)
	        face->set_boundary_id(1);
	      else
	        face->set_boundary_id(2);
	    }
	}

    m_triangulation.refine_global(m_parameters.m_global_refine_times);

    if (m_parameters.m_refinement_strategy == "pre-refine")
      {
	unsigned int material_id;
	double length_scale;
	for (unsigned int i = 0; i < m_parameters.m_local_prerefine_times; i++)
	  {
	    for (const auto &cell : m_triangulation.active_cell_iterators())
	      {
		if (   std::fabs(cell->center()[1]) < 0.01
		    && cell->center()[0] > 0.495)
		  {
		    material_id = cell->material_id();
		    length_scale = m_material_data[material_id][2];
		    if (  std::sqrt(cell->measure())
			> length_scale * m_parameters.m_allowed_max_h_l_ratio )
		      cell->set_refine_flag();
		  }
	      }
	    m_triangulation.execute_coarsening_and_refinement();
	  }
      }
    else if (m_parameters.m_refinement_strategy == "adaptive-refine")
      {
	unsigned int material_id;
	double length_scale;
	bool initiation_point_refine_unfinished = true;
	while (initiation_point_refine_unfinished)
	  {
	    initiation_point_refine_unfinished = false;
	    for (const auto &cell : m_triangulation.active_cell_iterators())
	      {
		if (   std::fabs(cell->center()[1] - 0.0) < 0.05
		    && std::fabs(cell->center()[0] - 0.5) < 0.05)
		  {
		    material_id = cell->material_id();
		    length_scale = m_material_data[material_id][2];
		    if (  std::sqrt(cell->measure())
			> length_scale * m_parameters.m_allowed_max_h_l_ratio )
		      {
		        cell->set_refine_flag();
		        initiation_point_refine_unfinished = true;
		      }
		  }
	      }
	    m_triangulation.execute_coarsening_and_refinement();
	  }
      }
    else
      {
	AssertThrow(false,
	            ExcMessage("Selected mesh refinement strategy not implemented!"));
      }
  }

  template <int dim>
  void PhaseFieldSplitSolve<dim>::make_grid_case_2()
  {
    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;
    m_logfile << "\t\t\t\tSquare shear (unstructured)" << std::endl;
    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;

    AssertThrow(dim==2, ExcMessage("The dimension has to be 2D!"));

    GridIn<dim> gridin;
    gridin.attach_triangulation(m_triangulation);
    std::ifstream f("square_shear_unstructured.msh");
    gridin.read_msh(f);

    for (const auto &cell : m_triangulation.active_cell_iterators())
      for (const auto &face : cell->face_iterators())
	{
	  if (face->at_boundary() == true)
	    {
	      if (std::fabs(face->center()[1] + 0.5 ) < 1.0e-9 )
		face->set_boundary_id(0);
	      else if (std::fabs(face->center()[1] - 0.5 ) < 1.0e-9)
	        face->set_boundary_id(1);
	      else if (   (std::fabs(face->center()[0] - 0.0 ) < 1.0e-9)
		       || (std::fabs(face->center()[0] - 1.0 ) < 1.0e-9))
	        face->set_boundary_id(2);
	      else
	        face->set_boundary_id(3);
	    }
	}

    m_triangulation.refine_global(m_parameters.m_global_refine_times);

    if (m_parameters.m_refinement_strategy == "pre-refine")
      {
	unsigned int material_id;
	double length_scale;
	for (unsigned int i = 0; i < m_parameters.m_local_prerefine_times; i++)
	  {
	    for (const auto &cell : m_triangulation.active_cell_iterators())
	      {
		if (    (cell->center()[0] > 0.45)
		     && (cell->center()[1] < 0.05) )
		  {
		    material_id = cell->material_id();
		    length_scale = m_material_data[material_id][2];
		    if (  std::sqrt(cell->measure())
			> length_scale * m_parameters.m_allowed_max_h_l_ratio )
		      cell->set_refine_flag();
		  }
	      }
	    m_triangulation.execute_coarsening_and_refinement();
	  }
      }
    else if (m_parameters.m_refinement_strategy == "adaptive-refine")
      {
	unsigned int material_id;
	double length_scale;
	bool initiation_point_refine_unfinished = true;
	while (initiation_point_refine_unfinished)
	  {
	    initiation_point_refine_unfinished = false;
	    for (const auto &cell : m_triangulation.active_cell_iterators())
	      {
		if (    std::fabs(cell->center()[0] - 0.5) < 0.025
		     && cell->center()[1] < 0.0 && cell->center()[1] > -0.025)
		  {
		    material_id = cell->material_id();
		    length_scale = m_material_data[material_id][2];
		    if (  std::sqrt(cell->measure())
			> length_scale * m_parameters.m_allowed_max_h_l_ratio )
		      {
		        cell->set_refine_flag();
		        initiation_point_refine_unfinished = true;
		      }
		  }
	      }
	    m_triangulation.execute_coarsening_and_refinement();
	  }
      }
    else
      {
	AssertThrow(false,
	            ExcMessage("Selected mesh refinement strategy not implemented!"));
      }
  }

  template <int dim>
  void PhaseFieldSplitSolve<dim>::make_grid_case_3()
  {
    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;
    m_logfile << "\t\t\tSquare tension (structured)" << std::endl;
    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;

    AssertThrow(dim==2, ExcMessage("The dimension has to be 2D!"));

    GridIn<dim> gridin;
    gridin.attach_triangulation(m_triangulation);
    std::ifstream f("square_tension_structured.msh");
    gridin.read_msh(f);

    for (const auto &cell : m_triangulation.active_cell_iterators())
      for (const auto &face : cell->face_iterators())
	{
	  if (face->at_boundary() == true)
	    {
	      if (std::fabs(face->center()[1] - 0.0 ) < 1.0e-9 )
		face->set_boundary_id(0);
	      else if (std::fabs(face->center()[1] - 1.0 ) < 1.0e-9)
	        face->set_boundary_id(1);
	      else
	        face->set_boundary_id(2);
	    }
	}

    m_triangulation.refine_global(m_parameters.m_global_refine_times);

    if (m_parameters.m_refinement_strategy == "pre-refine")
      {
	unsigned int material_id;
	double length_scale;
	for (unsigned int i = 0; i < m_parameters.m_local_prerefine_times; i++)
	  {
	    for (const auto &cell : m_triangulation.active_cell_iterators())
	      {
		if (    (std::fabs(cell->center()[1] - 0.5) < 0.025)
		     && (cell->center()[0] > 0.475) )
		  {
		    material_id = cell->material_id();
		    length_scale = m_material_data[material_id][2];
		    if (  std::sqrt(cell->measure())
			> length_scale * m_parameters.m_allowed_max_h_l_ratio )
		      cell->set_refine_flag();
		  }
	      }
	    m_triangulation.execute_coarsening_and_refinement();
	  }
      }
    else if (m_parameters.m_refinement_strategy == "adaptive-refine")
      {
	unsigned int material_id;
	double length_scale;
	bool initiation_point_refine_unfinished = true;
	while (initiation_point_refine_unfinished)
	  {
	    initiation_point_refine_unfinished = false;
	    for (const auto &cell : m_triangulation.active_cell_iterators())
	      {
		if (    std::fabs(cell->center()[0] - 0.5) < 0.025
		     && std::fabs(cell->center()[1] - 0.5) < 0.025 )
		  {
		    material_id = cell->material_id();
		    length_scale = m_material_data[material_id][2];
		    if (  std::sqrt(cell->measure())
			> length_scale * m_parameters.m_allowed_max_h_l_ratio )
		      {
		        cell->set_refine_flag();
		        initiation_point_refine_unfinished = true;
		      }
		  }
	      }
	    m_triangulation.execute_coarsening_and_refinement();
	  }
      }
    else
      {
	AssertThrow(false,
	            ExcMessage("Selected mesh refinement strategy not implemented!"));
      }
  }

  template <int dim>
  void PhaseFieldSplitSolve<dim>::make_grid_case_4()
  {
    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;
    m_logfile << "\t\t\t\tSquare shear (structured)" << std::endl;
    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;

    AssertThrow(dim==2, ExcMessage("The dimension has to be 2D!"));

    GridIn<dim> gridin;
    gridin.attach_triangulation(m_triangulation);
    std::ifstream f("square_shear_structured.msh");
    gridin.read_msh(f);

    for (const auto &cell : m_triangulation.active_cell_iterators())
      for (const auto &face : cell->face_iterators())
	{
	  if (face->at_boundary() == true)
	    {
	      if (std::fabs(face->center()[1] - 0.0 ) < 1.0e-9 )
		face->set_boundary_id(0);
	      else if (std::fabs(face->center()[1] - 1.0 ) < 1.0e-9)
	        face->set_boundary_id(1);
	      else if (   (std::fabs(face->center()[0] - 0.0 ) < 1.0e-9)
		       || (std::fabs(face->center()[0] - 1.0 ) < 1.0e-9))
	        face->set_boundary_id(2);
	      else
	        face->set_boundary_id(3);
	    }
	}

    m_triangulation.refine_global(m_parameters.m_global_refine_times);

    if (m_parameters.m_refinement_strategy == "pre-refine")
      {
	unsigned int material_id;
	double length_scale;
	for (unsigned int i = 0; i < m_parameters.m_local_prerefine_times; i++)
	  {
	    for (const auto &cell : m_triangulation.active_cell_iterators())
	      {
		if (    (cell->center()[0] > 0.475)
		     && (cell->center()[1] < 0.525) )
		  {
		    material_id = cell->material_id();
		    length_scale = m_material_data[material_id][2];
		    if (  std::sqrt(cell->measure())
			> length_scale * m_parameters.m_allowed_max_h_l_ratio )
		      cell->set_refine_flag();
		  }
	      }
	    m_triangulation.execute_coarsening_and_refinement();
	  }
      }
    else if (m_parameters.m_refinement_strategy == "adaptive-refine")
      {
	unsigned int material_id;
	double length_scale;
	bool initiation_point_refine_unfinished = true;
	while (initiation_point_refine_unfinished)
	  {
	    initiation_point_refine_unfinished = false;
	    for (const auto &cell : m_triangulation.active_cell_iterators())
	      {
		if (    std::fabs(cell->center()[0] - 0.5) < 0.025
		     && cell->center()[1] < 0.5 && cell->center()[1] > 0.475 )
		  {
		    material_id = cell->material_id();
		    length_scale = m_material_data[material_id][2];
		    if (  std::sqrt(cell->measure())
			> length_scale * m_parameters.m_allowed_max_h_l_ratio )
		      {
		        cell->set_refine_flag();
		        initiation_point_refine_unfinished = true;
		      }
		  }
	      }
	    m_triangulation.execute_coarsening_and_refinement();
	  }
      }
    else
      {
	AssertThrow(false,
	            ExcMessage("Selected mesh refinement strategy not implemented!"));
      }
  }



  template <int dim>
  void PhaseFieldSplitSolve<dim>::make_grid_case_9()
  {
    AssertThrow(dim==2, ExcMessage("The dimension has to be 2D!"));

    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;
    m_logfile << "\t\t\t\tL-shape bending (2D structured)" << std::endl;
    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;

    GridIn<dim> gridin;
    gridin.attach_triangulation(m_triangulation);
    std::ifstream f("L-Shape.msh");
    gridin.read_msh(f);

    for (const auto &cell : m_triangulation.active_cell_iterators())
      for (const auto &face : cell->face_iterators())
	{
	  if (face->at_boundary() == true)
	    {
	      if (std::fabs(face->center()[1] - 0.0 ) < 1.0e-9 )
		face->set_boundary_id(0);
	      else
	        face->set_boundary_id(1);
	    }
	}

    m_triangulation.refine_global(m_parameters.m_global_refine_times);

    if (m_parameters.m_refinement_strategy == "pre-refine")
      {
	unsigned int material_id;
	double length_scale;
	for (unsigned int i = 0; i < m_parameters.m_local_prerefine_times; i++)
	  {
	    for (const auto &cell : m_triangulation.active_cell_iterators())
	      {
		if (    (cell->center()[1] > 242.0)
		     && (cell->center()[1] < 312.5)
		     && (cell->center()[0] < 258.0) )
		  {
		    material_id = cell->material_id();
		    length_scale = m_material_data[material_id][2];
		    if (  std::sqrt(cell->measure())
			> length_scale * m_parameters.m_allowed_max_h_l_ratio )
		      cell->set_refine_flag();
		  }
	      }
	    m_triangulation.execute_coarsening_and_refinement();
	  }
      }
    else if (m_parameters.m_refinement_strategy == "adaptive-refine")
      {
	unsigned int material_id;
	double length_scale;
	bool initiation_point_refine_unfinished = true;
	while (initiation_point_refine_unfinished)
	  {
	    initiation_point_refine_unfinished = false;
	    for (const auto &cell : m_triangulation.active_cell_iterators())
	      {
		if (             (cell->center()[0] - 250) < 0.0
		     &&          (cell->center()[0] - 240) > 0.0
		     && std::fabs(cell->center()[1] - 250) < 10.0 )
		  {
		    material_id = cell->material_id();
		    length_scale = m_material_data[material_id][2];
		    if (  std::sqrt(cell->measure())
			> length_scale * m_parameters.m_allowed_max_h_l_ratio )
		      {
		        cell->set_refine_flag();
		        initiation_point_refine_unfinished = true;
		      }
		  }
	      }
	    m_triangulation.execute_coarsening_and_refinement();
	  }
      }
    else
      {
	AssertThrow(false,
	            ExcMessage("Selected mesh refinement strategy not implemented!"));
      }
  }

  template <int dim>
  void PhaseFieldSplitSolve<dim>::setup_system()
  {
    m_timer.enter_subsection("Setup system");

    setup_system_displacement();

    setup_system_phasefield();

    m_logfile << "\t\tTriangulation:"
              << "\n\t\t\t Number of active cells: "
              << m_triangulation.n_active_cells()
              << "\n\t\t\t Number of used vertices: "
              << m_triangulation.n_used_vertices()
              << "\n\t\t\t Number of active edges: "
              << m_triangulation.n_active_lines()
              << "\n\t\t\t Number of active faces: "
              << m_triangulation.n_active_faces()
              << "\n\t\t\t Number of degrees of freedom (total): "
	      << m_dof_handler_displacement.n_dofs()
	       + m_dof_handler_phasefield.n_dofs()
              << "\n\t\t\t Number of degrees of freedom (displacement): "
	      << m_dof_handler_displacement.n_dofs()
	      << "\n\t\t\t Number of degrees of freedom (phasefield): "
	      << m_dof_handler_phasefield.n_dofs()
              << std::endl;

    setup_qph();

    m_timer.leave_subsection();
  }

  template <int dim>
  void PhaseFieldSplitSolve<dim>::setup_system_phasefield()
  {
    m_dof_handler_phasefield.distribute_dofs(m_fe_phasefield);
    m_solution_phasefield.reinit(m_dof_handler_phasefield.n_dofs());
    m_system_rhs_phasefield.reinit(m_dof_handler_phasefield.n_dofs());

    m_constraints_phasefield.clear();
    DoFTools::make_hanging_node_constraints(m_dof_handler_phasefield, m_constraints_phasefield);
    m_constraints_phasefield.close();

    DynamicSparsityPattern dsp(m_dof_handler_phasefield.n_dofs(), m_dof_handler_phasefield.n_dofs());
    DoFTools::make_sparsity_pattern(m_dof_handler_phasefield,
                                    dsp,
                                    m_constraints_phasefield,
                                    /*keep_constrained_dofs = */ false);
    m_sparsity_pattern_phasefield.copy_from(dsp);

    m_system_matrix_phasefield.reinit(m_sparsity_pattern_phasefield);

    m_logfile << "\t\tTriangulation (phase-field):"
              << "\n\t\t\t Number of active cells: "
              << m_triangulation.n_active_cells()
              << "\n\t\t\t Number of used vertices: "
              << m_triangulation.n_used_vertices()
              << "\n\t\t\t Number of degrees of freedom (phase-field): "
	      << m_dof_handler_phasefield.n_dofs()
              << std::endl;
  }

  template <int dim>
  void PhaseFieldSplitSolve<dim>::setup_system_displacement()
  {
    m_dof_handler_displacement.distribute_dofs(m_fe_displacement);
    m_solution_displacement.reinit(m_dof_handler_displacement.n_dofs());
    m_system_rhs_displacement.reinit(m_dof_handler_displacement.n_dofs());

    m_constraints_displacement.clear();
    DoFTools::make_hanging_node_constraints(m_dof_handler_displacement, m_constraints_displacement);
    m_constraints_displacement.close();

    DynamicSparsityPattern dsp(m_dof_handler_displacement.n_dofs(),
			       m_dof_handler_displacement.n_dofs());
    DoFTools::make_sparsity_pattern(m_dof_handler_displacement,
                                    dsp,
                                    m_constraints_displacement,
                                    /*keep_constrained_dofs = */ false);
    m_sparsity_pattern_displacement.copy_from(dsp);

    m_system_matrix_displacement.reinit(m_sparsity_pattern_displacement);
  }

  template <int dim>
  void PhaseFieldSplitSolve<dim>::make_constraints_phasefield()
  {
    m_constraints_phasefield.clear();
    DoFTools::make_hanging_node_constraints(m_dof_handler_phasefield, m_constraints_phasefield);
    m_constraints_phasefield.close();
  }

  template <int dim>
  void PhaseFieldSplitSolve<dim>::make_constraints_displacement(const unsigned int it_nr,
								const unsigned int itr_stagger)
  {
    // The staggered iteration starts from 1
    const bool apply_dirichlet_bc = ( (it_nr == 0) && (itr_stagger == 1) );

    if ( (it_nr > 1) || (itr_stagger > 1) )
      {
        //m_logfile << " --- " << std::flush;
        return;
      }

    //m_logfile << " CST " << std::flush;

    if (apply_dirichlet_bc)
      {
	m_constraints_displacement.clear();
	DoFTools::make_hanging_node_constraints(m_dof_handler_displacement,
						m_constraints_displacement);
	const FEValuesExtractors::Scalar x_displacement(0);
	const FEValuesExtractors::Scalar y_displacement(1);
	const FEValuesExtractors::Scalar z_displacement(2);

	if (m_parameters.m_scenario == 1)
	  {
	    // Dirichlet B,C. bottom surface
	    const int boundary_id_bottom_surface = 0;
	    VectorTools::interpolate_boundary_values(m_dof_handler_displacement,
						     boundary_id_bottom_surface,
						     Functions::ZeroFunction<dim>(dim),
						     m_constraints_displacement,
						     m_fe_displacement.component_mask(y_displacement));

	    typename Triangulation<dim>::active_vertex_iterator vertex_itr;
	    vertex_itr = m_triangulation.begin_active_vertex();
	    std::vector<types::global_dof_index> node_xy(m_fe_displacement.dofs_per_vertex);

	    for (; vertex_itr != m_triangulation.end_vertex(); ++vertex_itr)
	      {
		if (   (std::fabs(vertex_itr->vertex()[0] - 0.0) < 1.0e-9)
		    && (std::fabs(vertex_itr->vertex()[1] + 0.5) < 1.0e-9) )
		  {
		    node_xy = usr_utilities::get_vertex_dofs(vertex_itr, m_dof_handler_displacement);
		  }
	      }
	    m_constraints_displacement.add_line(node_xy[0]);
	    m_constraints_displacement.set_inhomogeneity(node_xy[0], 0.0);

	    m_constraints_displacement.add_line(node_xy[1]);
	    m_constraints_displacement.set_inhomogeneity(node_xy[1], 0.0);


	    const int boundary_id_top_surface = 1;
	    /*
	    VectorTools::interpolate_boundary_values(m_dof_handler_displacement,
						     boundary_id_top_surface,
						     Functions::ZeroFunction<dim>(dim),
						     m_constraints_displacement,
						     m_fe_displacement.component_mask(x_displacement));
            */

            const double time_inc = m_time.get_delta_t();
            double disp_magnitude = m_time.get_magnitude();;
            //if (itr_stagger > 1)
            //  disp_magnitude = 0.0;
	    VectorTools::interpolate_boundary_values(m_dof_handler_displacement,
						     boundary_id_top_surface,
						     Functions::ConstantFunction<dim>(
						       disp_magnitude*time_inc, dim),
						     m_constraints_displacement,
						     m_fe_displacement.component_mask(y_displacement));
	  }
	else if (m_parameters.m_scenario == 2)
	  {
	    // Dirichlet B,C. bottom surface
	    const int boundary_id_bottom_surface = 0;
	    VectorTools::interpolate_boundary_values(m_dof_handler_displacement,
						     boundary_id_bottom_surface,
						     Functions::ZeroFunction<dim>(dim),
						     m_constraints_displacement);

	    const int boundary_id_top_surface = 1;
	    VectorTools::interpolate_boundary_values(m_dof_handler_displacement,
						     boundary_id_top_surface,
						     Functions::ZeroFunction<dim>(dim),
						     m_constraints_displacement,
						     m_fe_displacement.component_mask(y_displacement));

	    const double time_inc = m_time.get_delta_t();
	    double disp_magnitude = m_time.get_magnitude();;
	    //if (itr_stagger > 1)
	    //  disp_magnitude = 0.0;
	    VectorTools::interpolate_boundary_values(m_dof_handler_displacement,
						     boundary_id_top_surface,
						     Functions::ConstantFunction<dim>(
						       disp_magnitude*time_inc, dim),
						     m_constraints_displacement,
						     m_fe_displacement.component_mask(x_displacement));

	    const int boundary_id_side_surfaces = 2;
	    VectorTools::interpolate_boundary_values(m_dof_handler_displacement,
						     boundary_id_side_surfaces,
						     Functions::ZeroFunction<dim>(dim),
						     m_constraints_displacement,
						     m_fe_displacement.component_mask(y_displacement));
	  }
	else if (m_parameters.m_scenario == 3)
	  {
	    // Dirichlet B,C. bottom surface
	    const int boundary_id_bottom_surface = 0;
	    VectorTools::interpolate_boundary_values(m_dof_handler_displacement,
						     boundary_id_bottom_surface,
						     Functions::ZeroFunction<dim>(dim),
						     m_constraints_displacement,
						     m_fe_displacement.component_mask(y_displacement));

	    typename Triangulation<dim>::active_vertex_iterator vertex_itr;
	    vertex_itr = m_triangulation.begin_active_vertex();
	    std::vector<types::global_dof_index> node_xy(m_fe_displacement.dofs_per_vertex);

	    for (; vertex_itr != m_triangulation.end_vertex(); ++vertex_itr)
	      {
		if (   (std::fabs(vertex_itr->vertex()[0] - 0.0) < 1.0e-9)
		    && (std::fabs(vertex_itr->vertex()[1] - 0.0) < 1.0e-9) )
		  {
		    node_xy = usr_utilities::get_vertex_dofs(vertex_itr, m_dof_handler_displacement);
		  }
	      }
	    m_constraints_displacement.add_line(node_xy[0]);
	    m_constraints_displacement.set_inhomogeneity(node_xy[0], 0.0);

	    m_constraints_displacement.add_line(node_xy[1]);
	    m_constraints_displacement.set_inhomogeneity(node_xy[1], 0.0);


	    const int boundary_id_top_surface = 1;
	    /*
	    VectorTools::interpolate_boundary_values(m_dof_handler_displacement,
						     boundary_id_top_surface,
						     Functions::ZeroFunction<dim>(dim),
						     m_constraints_displacement,
						     m_fe_displacement.component_mask(x_displacement));
            */

            const double time_inc = m_time.get_delta_t();
            double disp_magnitude = m_time.get_magnitude();
            //if (itr_stagger > 1)
            //  disp_magnitude = 0.0;
	    VectorTools::interpolate_boundary_values(m_dof_handler_displacement,
						     boundary_id_top_surface,
						     Functions::ConstantFunction<dim>(
						       disp_magnitude*time_inc, dim),
						     m_constraints_displacement,
						     m_fe_displacement.component_mask(y_displacement));
	  }
	else if (m_parameters.m_scenario == 4)
	  {
	    // Dirichlet B,C. bottom surface
	    const int boundary_id_bottom_surface = 0;
	    VectorTools::interpolate_boundary_values(m_dof_handler_displacement,
						     boundary_id_bottom_surface,
						     Functions::ZeroFunction<dim>(dim),
						     m_constraints_displacement);

	    const int boundary_id_top_surface = 1;
	    VectorTools::interpolate_boundary_values(m_dof_handler_displacement,
						     boundary_id_top_surface,
						     Functions::ZeroFunction<dim>(dim),
						     m_constraints_displacement,
						     m_fe_displacement.component_mask(y_displacement));

	    const double time_inc = m_time.get_delta_t();
	    double disp_magnitude = m_time.get_magnitude();;
	    //if (itr_stagger > 1)
	    //  disp_magnitude = 0.0;
	    VectorTools::interpolate_boundary_values(m_dof_handler_displacement,
						     boundary_id_top_surface,
						     Functions::ConstantFunction<dim>(
						       disp_magnitude*time_inc, dim),
						     m_constraints_displacement,
						     m_fe_displacement.component_mask(x_displacement));

	    const int boundary_id_side_surfaces = 2;
	    VectorTools::interpolate_boundary_values(m_dof_handler_displacement,
						     boundary_id_side_surfaces,
						     Functions::ZeroFunction<dim>(dim),
						     m_constraints_displacement,
						     m_fe_displacement.component_mask(y_displacement));
	  }
	else if (m_parameters.m_scenario == 9)
	  {
	    // Dirichlet B,C. bottom surface
	    const int boundary_id_bottom_surface = 0;
	    VectorTools::interpolate_boundary_values(m_dof_handler_displacement,
						     boundary_id_bottom_surface,
						     Functions::ZeroFunction<dim>(dim),
						     m_constraints_displacement);

	    typename Triangulation<dim>::active_vertex_iterator vertex_itr;
	    vertex_itr = m_triangulation.begin_active_vertex();
	    std::vector<types::global_dof_index> node_disp_control(m_fe_displacement.dofs_per_vertex);

	    for (; vertex_itr != m_triangulation.end_vertex(); ++vertex_itr)
	      {
		if (   (std::fabs(vertex_itr->vertex()[0] - 470.0) < 1.0e-9)
		    && (std::fabs(vertex_itr->vertex()[1] - 250.0) < 1.0e-9) )
		  {
		    node_disp_control = usr_utilities::get_vertex_dofs(vertex_itr, m_dof_handler_displacement);
	            // node applied with y-displacement
		    const double time_inc = m_time.get_delta_t();
		    double disp_magnitude = m_time.get_magnitude();

		    m_constraints_displacement.add_line(node_disp_control[1]);
		    m_constraints_displacement.set_inhomogeneity(node_disp_control[1], disp_magnitude*time_inc);
		  }
	      }
	  }
	else
	  Assert(false, ExcMessage("The scenario has not been implemented!"));
      }
    else  // inhomogeneous constraints
      {
        if (m_constraints_displacement.has_inhomogeneities())
          {
            AffineConstraints<double> homogeneous_constraints(m_constraints_displacement);
            for (unsigned int dof = 0; dof != m_dof_handler_displacement.n_dofs(); ++dof)
              if (homogeneous_constraints.is_inhomogeneously_constrained(dof))
                homogeneous_constraints.set_inhomogeneity(dof, 0.0);

            m_constraints_displacement.clear();
            m_constraints_displacement.copy_from(homogeneous_constraints);
          }
      }
    m_constraints_displacement.close();
  }

  template <int dim>
  void PhaseFieldSplitSolve<dim>::assemble_system_one_cell_phasefield(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    ScratchData_ASM_phasefield & scratch,
    PerTaskData_ASM_phasefield & data) const
  {
    data.reset();
    scratch.reset();
    scratch.m_fe_values.reinit(cell);
    cell->get_dof_indices(data.m_local_dof_indices);

    scratch.m_fe_values.get_function_values(
      scratch.m_solution_previous_phasefield, scratch.m_old_phasefield);

    const std::vector<std::shared_ptr<const PointHistory<dim>>> lqph =
      m_quadrature_point_history.get_data(cell);
    Assert(lqph.size() == m_n_q_points, ExcInternalError());

    const double delta_time = m_time.get_delta_t();

    const FEValuesExtractors::Scalar phase_field(0);

    for (const unsigned int q_point :
         scratch.m_fe_values.quadrature_point_indices())
      {
        for (const unsigned int k : scratch.m_fe_values.dof_indices())
          {
	    scratch.m_Nx[q_point][k] =
	      scratch.m_fe_values[phase_field].value(k, q_point);
	    scratch.m_grad_Nx[q_point][k] =
	      scratch.m_fe_values[phase_field].gradient(k, q_point);
          }
      }

    for (unsigned int q_point : scratch.m_fe_values.quadrature_point_indices())
      {
	const double length_scale          = lqph[q_point]->get_length_scale();
	const double gc                    = lqph[q_point]->get_critical_energy_release_rate();
	const double eta                   = lqph[q_point]->get_viscosity();
	const double history_strain_energy = lqph[q_point]->get_history_max_positive_strain_energy();
	const double current_positive_strain_energy = lqph[q_point]->get_current_positive_strain_energy();

	double history_value = history_strain_energy;
	if (current_positive_strain_energy > history_strain_energy)
	  history_value = current_positive_strain_energy;

        const std::vector<double>         &      N = scratch.m_Nx[q_point];
        const std::vector<Tensor<1, dim>> & grad_N = scratch.m_grad_Nx[q_point];
        const double                old_phasefield = scratch.m_old_phasefield[q_point];
        const double                           JxW = scratch.m_fe_values.JxW(q_point);

        for (unsigned int i : scratch.m_fe_values.dof_indices())
          {
	    for (unsigned int j : scratch.m_fe_values.dof_indices_ending_at(i))
	      {
		data.m_cell_matrix(i, j) += (  (gc/length_scale + eta/delta_time + 2.0*history_value)
		                             * N[i] * N[j]
				             + gc * length_scale * grad_N[i] * grad_N[j] )
				           * JxW;
	      } // j
	    data.m_cell_rhs(i) += (  eta/delta_time * N[i] * old_phasefield
		                   + 2.0 * N[i] * history_value ) * JxW;
          } // i
      } // q_point

    for (const unsigned int i : scratch.m_fe_values.dof_indices())
      for (const unsigned int j : scratch.m_fe_values.dof_indices_starting_at(i + 1))
        data.m_cell_matrix(i, j) = data.m_cell_matrix(j, i);
  }

  template <int dim>
  void PhaseFieldSplitSolve<dim>::assemble_system_phasefield()
  {
    m_timer.enter_subsection("Assemble phase field system");

    m_system_matrix_phasefield = 0.0;
    m_system_rhs_phasefield    = 0.0;

    const UpdateFlags uf_cell(update_values | update_gradients |
			      update_JxW_values);

    PerTaskData_ASM_phasefield per_task_data(m_fe_phasefield.n_dofs_per_cell());
    ScratchData_ASM_phasefield scratch_data(m_fe_phasefield, m_qf_cell,
					    uf_cell,
					    m_solution_previous_timestep_phasefield);

    auto worker =
      [this](const typename DoFHandler<dim>::active_cell_iterator &cell,
	     ScratchData_ASM_phasefield &                                    scratch,
	     PerTaskData_ASM_phasefield &                                    data)
      {
        this->assemble_system_one_cell_phasefield(cell, scratch, data);
      };

    auto copier = [this](const PerTaskData_ASM_phasefield &data)
      {
        this->m_constraints_phasefield.distribute_local_to_global(data.m_cell_matrix,
                                                                  data.m_cell_rhs,
                                                                  data.m_local_dof_indices,
                                                                  m_system_matrix_phasefield,
                                                                  m_system_rhs_phasefield);
      };

    WorkStream::run(
      m_dof_handler_phasefield.active_cell_iterators(),
      worker,
      copier,
      scratch_data,
      per_task_data);

    m_timer.leave_subsection();
  }

  template <int dim>
  void PhaseFieldSplitSolve<dim>::assemble_rhs_phasefield()
  {
    m_timer.enter_subsection("Calculate phase-field residual");

    m_system_rhs_phasefield    = 0.0;

    const UpdateFlags uf_cell(update_values | update_gradients |
			      update_JxW_values);

    PerTaskData_RHS_phasefield per_task_data(m_fe_phasefield.n_dofs_per_cell());
    ScratchData_RHS_phasefield scratch_data(m_fe_phasefield, m_qf_cell,
					    uf_cell,
					    m_solution_previous_timestep_phasefield);

    auto worker =
      [this](const typename DoFHandler<dim>::active_cell_iterator &cell,
	     ScratchData_RHS_phasefield &                                    scratch,
	     PerTaskData_RHS_phasefield &                                    data)
      {
        this->assemble_rhs_one_cell_phasefield(cell, scratch, data);
      };

    auto copier = [this](const PerTaskData_RHS_phasefield &data)
      {
        this->m_constraints_phasefield.distribute_local_to_global(data.m_cell_rhs,
                                                                  data.m_local_dof_indices,
                                                                  m_system_rhs_phasefield);
      };

    WorkStream::run(
      m_dof_handler_phasefield.active_cell_iterators(),
      worker,
      copier,
      scratch_data,
      per_task_data);

    m_timer.leave_subsection();
  }


  template <int dim>
  void PhaseFieldSplitSolve<dim>::assemble_rhs_one_cell_phasefield(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    ScratchData_RHS_phasefield & scratch,
    PerTaskData_RHS_phasefield & data) const
  {
    data.reset();
    scratch.reset();
    scratch.m_fe_values.reinit(cell);
    cell->get_dof_indices(data.m_local_dof_indices);

    scratch.m_fe_values.get_function_values(
      scratch.m_solution_previous_phasefield, scratch.m_old_phasefield);

    const std::vector<std::shared_ptr<const PointHistory<dim>>> lqph =
      m_quadrature_point_history.get_data(cell);
    Assert(lqph.size() == m_n_q_points, ExcInternalError());

    const double delta_time = m_time.get_delta_t();

    const FEValuesExtractors::Scalar phase_field(0);

    for (const unsigned int q_point :
         scratch.m_fe_values.quadrature_point_indices())
      {
        for (const unsigned int k : scratch.m_fe_values.dof_indices())
          {
	    scratch.m_Nx[q_point][k] =
	      scratch.m_fe_values[phase_field].value(k, q_point);
	    scratch.m_grad_Nx[q_point][k] =
	      scratch.m_fe_values[phase_field].gradient(k, q_point);
          }
      }

    for (unsigned int q_point : scratch.m_fe_values.quadrature_point_indices())
      {
	const double length_scale          = lqph[q_point]->get_length_scale();
	const double gc                    = lqph[q_point]->get_critical_energy_release_rate();
	const double eta                   = lqph[q_point]->get_viscosity();
	const double history_strain_energy = lqph[q_point]->get_history_max_positive_strain_energy();
	const double current_positive_strain_energy = lqph[q_point]->get_current_positive_strain_energy();

	double history_value = history_strain_energy;
	if (current_positive_strain_energy > history_strain_energy)
	  history_value = current_positive_strain_energy;

	const double phasefield_value        = lqph[q_point]->get_phase_field_value();
	const Tensor<1, dim> phasefield_grad = lqph[q_point]->get_phase_field_gradient();

        const std::vector<double>         &      N = scratch.m_Nx[q_point];
        const std::vector<Tensor<1, dim>> & grad_N = scratch.m_grad_Nx[q_point];
        const double                old_phasefield = scratch.m_old_phasefield[q_point];
        const double                           JxW = scratch.m_fe_values.JxW(q_point);

        for (unsigned int i : scratch.m_fe_values.dof_indices())
          {
	    data.m_cell_rhs(i) += (    gc * length_scale * grad_N[i] * phasefield_grad
				    +  (   gc / length_scale * phasefield_value
					 + eta / delta_time  * (phasefield_value - old_phasefield)
					 + degradation_function_derivative(phasefield_value) * history_value )
				      * N[i]
				  ) * JxW;
          } // i
      } // q_point
  }

  template <int dim>
  void PhaseFieldSplitSolve<dim>::assemble_system_displacement()
  {
    m_timer.enter_subsection("Assemble displacement system");

    //m_logfile << " ASM_SYS " << std::flush;

    m_system_matrix_displacement = 0.0;
    m_system_rhs_displacement    = 0.0;

    const UpdateFlags uf_cell(update_values | update_gradients |
			      update_quadrature_points | update_JxW_values);
    const UpdateFlags uf_face(update_values | update_normal_vectors |
                              update_JxW_values);

    PerTaskData_ASM_displacement per_task_data(m_fe_displacement.n_dofs_per_cell());
    ScratchData_ASM_displacement scratch_data(m_fe_displacement, m_qf_cell, uf_cell, m_qf_face, uf_face);

    auto worker =
      [this](const typename DoFHandler<dim>::active_cell_iterator &cell,
	     ScratchData_ASM_displacement & scratch,
	     PerTaskData_ASM_displacement & data)
      {
        this->assemble_system_one_cell_displacement(cell, scratch, data);
      };

    auto copier = [this](const PerTaskData_ASM_displacement &data)
      {
        this->m_constraints_displacement.distribute_local_to_global(data.m_cell_matrix,
                                                                    data.m_cell_rhs,
                                                                    data.m_local_dof_indices,
								    m_system_matrix_displacement,
								    m_system_rhs_displacement);
      };

    WorkStream::run(
      m_dof_handler_displacement.active_cell_iterators(),
      worker,
      copier,
      scratch_data,
      per_task_data);

    m_timer.leave_subsection();
  }

  template <int dim>
  void PhaseFieldSplitSolve<dim>::assemble_system_one_cell_displacement(
      const typename DoFHandler<dim>::active_cell_iterator &cell,
      ScratchData_ASM_displacement & scratch,
      PerTaskData_ASM_displacement & data) const
  {
    data.reset();
    scratch.reset();
    scratch.m_fe_values.reinit(cell);
    cell->get_dof_indices(data.m_local_dof_indices);

    const std::vector<std::shared_ptr<const PointHistory<dim>>> lqph =
      m_quadrature_point_history.get_data(cell);
    Assert(lqph.size() == m_n_q_points, ExcInternalError());

    const double time_ramp = (m_time.current() / m_time.end());
    std::vector<Tensor<1, dim>> rhs_values(m_n_q_points);

    right_hand_side(scratch.m_fe_values.get_quadrature_points(),
		    rhs_values,
		    m_parameters.m_x_component*1.0,
		    m_parameters.m_y_component*1.0,
		    m_parameters.m_z_component*1.0);

    const FEValuesExtractors::Vector displacement(0);

    for (const unsigned int q_point :
         scratch.m_fe_values.quadrature_point_indices())
      {
        for (const unsigned int k : scratch.m_fe_values.dof_indices())
          {
	    scratch.m_Nx[q_point][k] =
	      scratch.m_fe_values[displacement].value(k, q_point);
	    scratch.m_grad_Nx[q_point][k] =
	      scratch.m_fe_values[displacement].gradient(k, q_point);
	    scratch.m_symm_grad_Nx[q_point][k] =
	      symmetrize(scratch.m_grad_Nx[q_point][k]);
          }
      }

    for (const unsigned int q_point :
         scratch.m_fe_values.quadrature_point_indices())
      {
        const SymmetricTensor<2, dim> & cauchy_stress = lqph[q_point]->get_cauchy_stress();
        const SymmetricTensor<4, dim> & mechanical_C  = lqph[q_point]->get_mechanical_C();

        const std::vector<Tensor<1,dim>> & N = scratch.m_Nx[q_point];
        const std::vector<SymmetricTensor<2, dim>> & symm_grad_N =
          scratch.m_symm_grad_Nx[q_point];
        const double JxW = scratch.m_fe_values.JxW(q_point);

        SymmetricTensor<2, dim> symm_grad_Nx_i_x_C;

        for (const unsigned int i : scratch.m_fe_values.dof_indices())
          {
            data.m_cell_rhs(i) -= (symm_grad_N[i] * cauchy_stress) * JxW;
            // contributions from the body force to right-hand side
            data.m_cell_rhs(i) += N[i] * rhs_values[q_point] * JxW;

            symm_grad_Nx_i_x_C = symm_grad_N[i] * mechanical_C;
            for (const unsigned int j : scratch.m_fe_values.dof_indices_ending_at(i))
              {
        	data.m_cell_matrix(i, j) += symm_grad_Nx_i_x_C * symm_grad_N[j] * JxW;
              } // j
          }  // i
      }  // q_point

    // if there is surface pressure, this surface pressure always applied to the
    // reference configuration
    const unsigned int face_pressure_id = 100;
    const double p0 = 0.0;

    for (const auto &face : cell->face_iterators())
      if (face->at_boundary() && face->boundary_id() == face_pressure_id)
        {
          scratch.m_fe_face_values.reinit(cell, face);

          for (const unsigned int f_q_point :
               scratch.m_fe_face_values.quadrature_point_indices())
            {
              const Tensor<1, dim> &N =
                scratch.m_fe_face_values.normal_vector(f_q_point);

              const double         pressure  = p0 * time_ramp;
              const Tensor<1, dim> traction  = pressure * N;

              for (const unsigned int i : scratch.m_fe_values.dof_indices())
                {
		  const unsigned int component_i =
		    m_fe_displacement.system_to_component_index(i).first;
		  const double Ni =
		    scratch.m_fe_face_values.shape_value(i, f_q_point);
		  const double JxW = scratch.m_fe_face_values.JxW(f_q_point);

		  data.m_cell_rhs(i) += (Ni * traction[component_i]) * JxW;
                }
            }
        }

    for (const unsigned int i : scratch.m_fe_values.dof_indices())
      for (const unsigned int j : scratch.m_fe_values.dof_indices_starting_at(i + 1))
        data.m_cell_matrix(i, j) = data.m_cell_matrix(j, i);
  }


  template <int dim>
  void PhaseFieldSplitSolve<dim>::assemble_rhs_displacement()
  {
    m_timer.enter_subsection("Calculate displacement residual");

    m_system_rhs_displacement    = 0.0;

    const UpdateFlags uf_cell(update_values | update_gradients |
			      update_quadrature_points | update_JxW_values);
    const UpdateFlags uf_face(update_values | update_normal_vectors |
                              update_JxW_values);

    PerTaskData_RHS_displacement per_task_data(m_fe_displacement.n_dofs_per_cell());
    ScratchData_RHS_displacement scratch_data(m_fe_displacement, m_qf_cell, uf_cell, m_qf_face, uf_face);

    auto worker =
      [this](const typename DoFHandler<dim>::active_cell_iterator &cell,
	     ScratchData_RHS_displacement & scratch,
	     PerTaskData_RHS_displacement & data)
      {
        this->assemble_rhs_one_cell_displacement(cell, scratch, data);
      };

    auto copier = [this](const PerTaskData_RHS_displacement &data)
      {
        this->m_constraints_displacement.distribute_local_to_global(data.m_cell_rhs,
                                                                    data.m_local_dof_indices,
								    m_system_rhs_displacement);
      };

    WorkStream::run(
      m_dof_handler_displacement.active_cell_iterators(),
      worker,
      copier,
      scratch_data,
      per_task_data);

    m_timer.leave_subsection();
  }

  template <int dim>
  void PhaseFieldSplitSolve<dim>::assemble_rhs_one_cell_displacement(
      const typename DoFHandler<dim>::active_cell_iterator &cell,
      ScratchData_RHS_displacement & scratch,
      PerTaskData_RHS_displacement & data) const
  {
    data.reset();
    scratch.reset();
    scratch.m_fe_values.reinit(cell);
    cell->get_dof_indices(data.m_local_dof_indices);

    const std::vector<std::shared_ptr<const PointHistory<dim>>> lqph =
      m_quadrature_point_history.get_data(cell);
    Assert(lqph.size() == m_n_q_points, ExcInternalError());

    const double time_ramp = (m_time.current() / m_time.end());
    std::vector<Tensor<1, dim>> rhs_values(m_n_q_points);

    right_hand_side(scratch.m_fe_values.get_quadrature_points(),
		    rhs_values,
		    m_parameters.m_x_component*1.0,
		    m_parameters.m_y_component*1.0,
		    m_parameters.m_z_component*1.0);

    const FEValuesExtractors::Vector displacement(0);

    for (const unsigned int q_point :
         scratch.m_fe_values.quadrature_point_indices())
      {
        for (const unsigned int k : scratch.m_fe_values.dof_indices())
          {
	    scratch.m_Nx[q_point][k] =
	      scratch.m_fe_values[displacement].value(k, q_point);
	    scratch.m_grad_Nx[q_point][k] =
	      scratch.m_fe_values[displacement].gradient(k, q_point);
	    scratch.m_symm_grad_Nx[q_point][k] =
	      symmetrize(scratch.m_grad_Nx[q_point][k]);
          }
      }

    for (const unsigned int q_point :
         scratch.m_fe_values.quadrature_point_indices())
      {
        const SymmetricTensor<2, dim> & cauchy_stress = lqph[q_point]->get_cauchy_stress();

        const std::vector<Tensor<1,dim>> & N = scratch.m_Nx[q_point];
        const std::vector<SymmetricTensor<2, dim>> & symm_grad_N =
          scratch.m_symm_grad_Nx[q_point];
        const double JxW = scratch.m_fe_values.JxW(q_point);

        for (const unsigned int i : scratch.m_fe_values.dof_indices())
          {
            data.m_cell_rhs(i) += (symm_grad_N[i] * cauchy_stress) * JxW;
            // contributions from the body force to right-hand side
            data.m_cell_rhs(i) -= N[i] * rhs_values[q_point] * JxW;
          }  // i
      }  // q_point

    // if there is surface pressure, this surface pressure always applied to the
    // reference configuration
    const unsigned int face_pressure_id = 100;
    const double p0 = 0.0;

    for (const auto &face : cell->face_iterators())
      if (face->at_boundary() && face->boundary_id() == face_pressure_id)
        {
          scratch.m_fe_face_values.reinit(cell, face);

          for (const unsigned int f_q_point :
               scratch.m_fe_face_values.quadrature_point_indices())
            {
              const Tensor<1, dim> &N =
                scratch.m_fe_face_values.normal_vector(f_q_point);

              const double         pressure  = p0 * time_ramp;
              const Tensor<1, dim> traction  = pressure * N;

              for (const unsigned int i : scratch.m_fe_values.dof_indices())
                {
		  const unsigned int component_i =
		    m_fe_displacement.system_to_component_index(i).first;
		  const double Ni =
		    scratch.m_fe_face_values.shape_value(i, f_q_point);
		  const double JxW = scratch.m_fe_face_values.JxW(f_q_point);

		  data.m_cell_rhs(i) -= (Ni * traction[component_i]) * JxW;
                }
            }
        }
  }

  template <int dim>
  void PhaseFieldSplitSolve<dim>::update_history_field_step()
  {
    for (const auto &cell : m_triangulation.active_cell_iterators())
      {
        std::vector<std::shared_ptr< PointHistory<dim>>> lqph =
          m_quadrature_point_history.get_data(cell);
        Assert(lqph.size() == m_n_q_points, ExcInternalError());

        for (unsigned int q_point = 0; q_point < m_n_q_points; ++q_point)
          {
            lqph[q_point]->update_history_variable();
          }
      }
  }

  template <int dim>
  void PhaseFieldSplitSolve<dim>::solve_linear_system_phasefield()
  {
    m_timer.enter_subsection("Solve phase field linear system");

/*
    {
    SolverControl            solver_control(1e6, 1e-9);
    SolverCG<Vector<double>> cg(solver_control);
    cg.connect_condition_number_slot(
	[] (double condition_number)
	{
          m_logfile << "   Estimated condition number = "<< condition_number << std::endl;
	},
	false);

    PreconditionSSOR<SparseMatrix<double>> preconditioner;
    preconditioner.initialize(m_system_matrix_phasefield, 1.2);

    cg.solve(m_system_matrix_phasefield, m_solution_phasefield, m_system_rhs_phasefield, preconditioner);
    }
*/
    //m_logfile << "   " << solver_control.last_step()
     //           << " CG iterations needed to obtain convergence." << std::endl;

    if (m_parameters.m_type_linear_solver == "Direct")
      {
        SparseDirectUMFPACK A_direct;
        A_direct.initialize(m_system_matrix_phasefield);
        A_direct.vmult(m_solution_phasefield,
	               m_system_rhs_phasefield);
      }
    else if (m_parameters.m_type_linear_solver == "CG")
      {
	SolverControl            solver_control_phasefield(1e6, 1e-15);
	SolverCG<Vector<double>> cg_phasefield(solver_control_phasefield);

	PreconditionJacobi<SparseMatrix<double>> preconditioner_phasefield;
	preconditioner_phasefield.initialize(m_system_matrix_phasefield, 1.0);

	cg_phasefield.solve(m_system_matrix_phasefield,
			    m_solution_phasefield,
			    m_system_rhs_phasefield,
			    preconditioner_phasefield);
      }
    else
      {
	AssertThrow(false,
	            ExcMessage("Selected linear solver not implemented for the phase-field subproblem!"));
      }

    m_constraints_phasefield.distribute(m_solution_phasefield);

    m_timer.leave_subsection();
  }

  template <int dim>
  void PhaseFieldSplitSolve<dim>::phasefield_step()
  {
    make_constraints_phasefield();
    assemble_system_phasefield();
    solve_linear_system_phasefield();
    update_phasefield();
  }

  template <int dim>
  unsigned int PhaseFieldSplitSolve<dim>::displacement_step(unsigned int itr_stagger)
  {
    Vector<double> solution_delta_displacement(m_dof_handler_displacement.n_dofs());
    solution_delta_displacement = 0.0;
    unsigned int newton_itrs_required = 0;
    newton_itrs_required = solve_nonlinear_displacement_newton_raphson(solution_delta_displacement, itr_stagger);
    m_solution_displacement += solution_delta_displacement;
    return newton_itrs_required;
  }

  template <int dim>
  void PhaseFieldSplitSolve<dim>::solve_linear_system_displacement(Vector<double> & newton_update)
  {
    m_timer.enter_subsection("Solve displacement linear system");

    //m_logfile << " SLV " << std::flush;
    //std::vector<double> linear_solver_parameters(3);
/*
    {
      SolverControl            solver_control(1e6, 1e-9);
      SolverCG<Vector<double>> cg(solver_control);
      cg.connect_condition_number_slot(
	  [&] (double condition_number)
	  {
	    linear_solver_parameters[0] = condition_number;
	    //m_logfile << "   Estimated condition number = "<< condition_number << std::endl;
	  },
	  false);

      PreconditionSSOR<SparseMatrix<double>> preconditioner;
      preconditioner.initialize(m_system_matrix_displacement, 1.2);

      cg.solve(m_system_matrix_displacement,
	       newton_update,
	       m_system_rhs_displacement,
	       preconditioner);

      //m_logfile << "   " << solver_control.last_step()
		  //<< " CG iterations needed to obtain convergence." << std::endl;
      linear_solver_parameters[1] = solver_control.last_step();
      linear_solver_parameters[2] = solver_control.last_value();
    }
*/
    if (m_parameters.m_type_linear_solver == "Direct")
      {
        SparseDirectUMFPACK A_direct;
        A_direct.initialize(m_system_matrix_displacement);
        A_direct.vmult(newton_update,
	               m_system_rhs_displacement);
      }
    else if (m_parameters.m_type_linear_solver == "CG")
      {
	SolverControl            solver_control_displacement(1e6, 1e-9);
	SolverCG<Vector<double>> cg_displacement(solver_control_displacement);

	PreconditionJacobi<SparseMatrix<double>> preconditioner_displacement;
	preconditioner_displacement.initialize(m_system_matrix_displacement, 1.0);

	cg_displacement.solve(m_system_matrix_displacement,
			      newton_update,
			      m_system_rhs_displacement,
			      preconditioner_displacement);
      }
    else
      {
	AssertThrow(false,
	            ExcMessage("Selected linear solver not implemented for the phase-field subproblem!"));
      }

    m_constraints_displacement.distribute(newton_update);

    m_timer.leave_subsection();
    //return linear_solver_parameters;
  }

  template <int dim>
  void PhaseFieldSplitSolve<dim>::print_conv_header()
  {
    static const unsigned int l_width = 133;
    m_logfile << '\t';
    for (unsigned int i = 0; i < l_width; ++i)
      m_logfile << '_';
    m_logfile << std::endl;

    m_logfile << "\tAM-itr\t"
              << "Subproblem  "
              << "No. Newton    Res.       Inc.   "
              << "   Subproblem  "
	      << "      Res_u"
	      << "      Res_d"
	      << "      Inc_u"
	      << "      Inc_d"
	      << "      Energy"
	      << std::endl;

    m_logfile << '\t';
    for (unsigned int i = 0; i < l_width; ++i)
      m_logfile << '_';
    m_logfile << std::endl;
  }

  template <int dim>
  unsigned int PhaseFieldSplitSolve<dim>::
  solve_nonlinear_displacement_newton_raphson(Vector<double> & solution_delta_displacement,
					      unsigned int itr_stagger)
  {
    Vector<double> newton_update(m_dof_handler_displacement.n_dofs());

    m_error_residual_displacement.reset();
    m_error_update_displacement.reset();

    unsigned int newton_iteration = 0;
    for (; newton_iteration < m_parameters.m_max_iterations_newton; ++newton_iteration)
      {
        make_constraints_displacement(newton_iteration, itr_stagger);
        assemble_system_displacement();

        get_error_residual_displacement(m_error_residual_displacement);

        if (    newton_iteration > 0
	     && m_error_residual_displacement.m_norm <= 1.0e-9
             && m_error_update_displacement.m_norm <= 1.0e-6 )
          {
	    if (m_parameters.m_output_iteration_history)
	      m_logfile << "  \t" << newton_iteration << "\t"
			<< std::setprecision(3)
			<< std::setw(7)
			<< std::scientific
			<< m_error_residual_displacement.m_norm
			<< "  "
			<< m_error_update_displacement.m_norm
			<< std::flush;
            break;
          }

        solve_linear_system_displacement(newton_update);

        get_error_update_displacement(newton_update, m_error_update_displacement);

        solution_delta_displacement += newton_update;
        update_qph_incremental(solution_delta_displacement);
      }

    AssertThrow(newton_iteration < m_parameters.m_max_iterations_newton,
                ExcMessage("No convergence in nonlinear solver!"));

    return newton_iteration;
  }

  template <int dim>
  void PhaseFieldSplitSolve<dim>::output_results() const
  {
    m_timer.enter_subsection("Output results");

    DataOut<dim> data_out;
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation_mechanical(
        dim, DataComponentInterpretation::component_is_part_of_vector);
    std::vector<std::string> solution_name(dim, "displacement");

    data_out.add_data_vector(m_dof_handler_displacement,
	                     m_solution_displacement,
                             solution_name,
                             data_component_interpretation_mechanical);

    data_out.add_data_vector(m_dof_handler_phasefield,
			     m_solution_phasefield,
			     "phasefield");

    Vector<double> cell_material_id(m_triangulation.n_active_cells());
    // output material ID for each cell
    for (const auto &cell : m_triangulation.active_cell_iterators())
      {
	cell_material_id(cell->active_cell_index()) = cell->material_id();
      }
    data_out.add_data_vector(cell_material_id, "materialID");

    // Stress L2 projection
    DoFHandler<dim> stresses_dof_handler_L2(m_triangulation);
    FE_Q<dim>     stresses_fe_L2(m_parameters.m_poly_degree); //FE_Q element is continuous
    stresses_dof_handler_L2.distribute_dofs(stresses_fe_L2);
    AffineConstraints<double> constraints;
    constraints.clear();
    DoFTools::make_hanging_node_constraints(stresses_dof_handler_L2, constraints);
    constraints.close();
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
	  data_component_interpretation_stress(1,
					       DataComponentInterpretation::component_is_scalar);

    for (unsigned int i = 0; i < dim; ++i)
      for (unsigned int j = i; j < dim; ++j)
	{
	  Vector<double> stress_field_L2;
	  stress_field_L2.reinit(stresses_dof_handler_L2.n_dofs());

	  MappingQ<dim> mapping(m_parameters.m_poly_degree + 1);
	  VectorTools::project(mapping,
			       stresses_dof_handler_L2,
			       constraints,
			       m_qf_cell,
			       [&] (const typename DoFHandler<dim>::active_cell_iterator & cell,
				    const unsigned int q) -> double
			       {
				 return m_quadrature_point_history.get_data(cell)[q]->get_cauchy_stress()[i][j];
			       },
			       stress_field_L2);

	  std::string stress_name = "Cauchy_stress_" + std::to_string(i+1) + std::to_string(j+1)
				  + "_L2";

	  data_out.add_data_vector(stresses_dof_handler_L2,
				   stress_field_L2,
				   stress_name,
				   data_component_interpretation_stress);
	}

    data_out.build_patches(std::min(m_fe_displacement.degree, m_fe_phasefield.degree));

    std::ofstream output("Solution-" + std::to_string(dim) + "d-" +
			 Utilities::int_to_string(m_time.get_timestep(),4) + ".vtu");

    data_out.write_vtu(output);
    m_timer.leave_subsection();
  }


  template <int dim>
  void PhaseFieldSplitSolve<dim>::calculate_reaction_force(unsigned int face_ID)
  {
    m_timer.enter_subsection("Calculate reaction force");

    Vector<double>       system_rhs_displacement;
    system_rhs_displacement.reinit(m_dof_handler_displacement.n_dofs());

    const unsigned int   dofs_per_cell_displacement = m_fe_displacement.n_dofs_per_cell();
    Vector<double> cell_rhs(dofs_per_cell_displacement);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell_displacement);
    const double time_ramp = (m_time.current() / m_time.end());
    std::vector<Tensor<1, dim>> rhs_values(m_n_q_points);
    const UpdateFlags uf_cell(update_values | update_gradients |
			      update_quadrature_points | update_JxW_values);
    const UpdateFlags uf_face(update_values | update_normal_vectors |
                              update_JxW_values);

    FEValues<dim> fe_values(m_fe_displacement, m_qf_cell, uf_cell);
    FEFaceValues<dim> fe_face_values(m_fe_displacement, m_qf_face, uf_face);

    // shape function values for displacement field
    std::vector<std::vector<Tensor<1, dim>>>
      Nx(m_qf_cell.size(), std::vector<Tensor<1, dim>>(dofs_per_cell_displacement));
    std::vector<std::vector<Tensor<2, dim>>>
      grad_Nx(m_qf_cell.size(),
              std::vector<Tensor<2, dim>>(dofs_per_cell_displacement));
    std::vector<std::vector<SymmetricTensor<2, dim>>>
      symm_grad_Nx(m_qf_cell.size(),
                   std::vector<SymmetricTensor<2, dim>>(dofs_per_cell_displacement));

    for (const auto &cell : m_dof_handler_displacement.active_cell_iterators())
      {
	// if calculate_reaction_force() is defined as const, then
	// we also need to put a const in std::shared_ptr,
	// that is, std::shared_ptr<const PointHistory<dim>>
	const std::vector<std::shared_ptr< PointHistory<dim>>> lqph =
	  m_quadrature_point_history.get_data(cell);
	Assert(lqph.size() == m_n_q_points, ExcInternalError());
        cell_rhs = 0.0;
        fe_values.reinit(cell);
        right_hand_side(fe_values.get_quadrature_points(),
    		        rhs_values,
    		        m_parameters.m_x_component*time_ramp,
    		        m_parameters.m_y_component*time_ramp,
    		        m_parameters.m_z_component*time_ramp);

        const FEValuesExtractors::Vector displacement(0);

        for (const unsigned int q_point : fe_values.quadrature_point_indices())
          {
            for (const unsigned int k : fe_values.dof_indices())
              {
		Nx[q_point][k] = fe_values[displacement].value(k, q_point);
		grad_Nx[q_point][k] = fe_values[displacement].gradient(k, q_point);
		symm_grad_Nx[q_point][k] = symmetrize(grad_Nx[q_point][k]);
              }
          }

        for (const unsigned int q_point : fe_values.quadrature_point_indices())
          {
            const SymmetricTensor<2, dim> & cauchy_stress = lqph[q_point]->get_cauchy_stress();

            const std::vector<Tensor<1,dim>> & N = Nx[q_point];
            const std::vector<SymmetricTensor<2, dim>> & symm_grad_N = symm_grad_Nx[q_point];
            const double JxW = fe_values.JxW(q_point);

            for (const unsigned int i : fe_values.dof_indices())
              {
                cell_rhs(i) -= (symm_grad_N[i] * cauchy_stress) * JxW;

    		    // contributions from the body force to right-hand side
    		cell_rhs(i) += N[i] * rhs_values[q_point] * JxW;
              }
          }

        // if there is surface pressure, this surface pressure always applied to the
        // reference configuration
        const unsigned int face_pressure_id = 100;
        const double p0 = 0.0;

        for (const auto &face : cell->face_iterators())
          {
	    if (face->at_boundary() && face->boundary_id() == face_pressure_id)
	      {
		fe_face_values.reinit(cell, face);

		for (const unsigned int f_q_point : fe_face_values.quadrature_point_indices())
		  {
		    const Tensor<1, dim> &N = fe_face_values.normal_vector(f_q_point);

		    const double         pressure  = p0 * time_ramp;
		    const Tensor<1, dim> traction  = pressure * N;

		    for (const unsigned int i : fe_values.dof_indices())
		      {
			const unsigned int component_i =
			    m_fe_displacement.system_to_component_index(i).first;
			const double Ni = fe_face_values.shape_value(i, f_q_point);
			const double JxW = fe_face_values.JxW(f_q_point);

			cell_rhs(i) += (Ni * traction[component_i]) * JxW;
		      }
		  }
	      }
          }

        cell->get_dof_indices(local_dof_indices);
        for (const unsigned int i : fe_values.dof_indices())
          system_rhs_displacement(local_dof_indices[i]) += cell_rhs(i);
      } // for (const auto &cell : m_dof_handler_displacement.active_cell_iterators())

    std::vector< types::global_dof_index > mapping;
    std::set<types::boundary_id> boundary_ids;
    boundary_ids.insert(face_ID);
    DoFTools::map_dof_to_boundary_indices(m_dof_handler_displacement,
					  boundary_ids,
					  mapping);

    std::vector<double> reaction_force(dim, 0.0);

    // Assume that the displacement dofs are the first "m_dofs_per_block[m_u_dof]" dofs
    for (unsigned int i = 0; i < m_dof_handler_displacement.n_dofs(); ++i)
      {
	if (mapping[i] != numbers::invalid_dof_index)
	  {
	    reaction_force[i % dim] += system_rhs_displacement(i);
	  }
      }

    for (unsigned int i = 0; i < dim; i++)
      m_logfile << "\t\tReaction force in direction " << i << " on boundary ID " << face_ID
                << " = "
		<< std::fixed << std::setprecision(3) << std::setw(1)
                << std::scientific
		<< reaction_force[i] << std::endl;

    std::pair<double, std::vector<double>> time_force;
    time_force.first = m_time.current();
    time_force.second = reaction_force;
    m_history_reaction_force.push_back(time_force);

    m_timer.leave_subsection();
  }

  template <int dim>
  void PhaseFieldSplitSolve<dim>::write_history_data()
  {
    m_logfile << "\t\tWrite history data ... \n"<<std::endl;

    std::ofstream myfile_reaction_force ("Reaction_force.hist");
    if (myfile_reaction_force.is_open())
    {
      myfile_reaction_force << 0.0 << "\t";
      if (dim == 2)
	myfile_reaction_force << 0.0 << "\t"
	       << 0.0 << std::endl;
      if (dim == 3)
	myfile_reaction_force << 0.0 << "\t"
	       << 0.0 << "\t"
	       << 0.0 << std::endl;

      for (auto const time_force : m_history_reaction_force)
	{
	  myfile_reaction_force << time_force.first << "\t";
	  if (dim == 2)
	    myfile_reaction_force << time_force.second[0] << "\t"
	           << time_force.second[1] << std::endl;
	  if (dim == 3)
	    myfile_reaction_force << time_force.second[0] << "\t"
	           << time_force.second[1] << "\t"
		   << time_force.second[2] << std::endl;
	}
      myfile_reaction_force.close();
    }
    else
      m_logfile << "Unable to open file";

    std::ofstream myfile_energy ("Energy.hist");
    if (myfile_energy.is_open())
    {
      myfile_energy << std::fixed << std::setprecision(10) << std::scientific
                    << 0.0 << "\t"
                    << 0.0 << "\t"
	            << 0.0 << "\t"
	            << 0.0 << std::endl;

      for (auto const time_energy : m_history_energy)
	{
	  myfile_energy << std::fixed << std::setprecision(10) << std::scientific
	                << time_energy.first     << "\t"
                        << time_energy.second[0] << "\t"
	                << time_energy.second[1] << "\t"
		        << time_energy.second[2] << std::endl;
	}
      myfile_energy.close();
    }
    else
      m_logfile << "Unable to open file";
  }

  template <int dim>
  double PhaseFieldSplitSolve<dim>::calculate_energy_functional() const
  {
    double energy_functional = 0.0;

    FEValues<dim> fe_values(m_fe_phasefield, m_qf_cell, update_JxW_values);

    for (const auto &cell : m_dof_handler_phasefield.active_cell_iterators())
      {
        fe_values.reinit(cell);

        const std::vector<std::shared_ptr<const PointHistory<dim>>> lqph =
          m_quadrature_point_history.get_data(cell);
        Assert(lqph.size() == m_n_q_points, ExcInternalError());

        for (unsigned int q_point = 0; q_point < m_n_q_points; ++q_point)
          {
            const double JxW = fe_values.JxW(q_point);
            energy_functional += lqph[q_point]->get_total_strain_energy() * JxW;
            energy_functional += lqph[q_point]->get_crack_energy_dissipation() * JxW;
          }
      }

    return energy_functional;
  }

  template <int dim>
  std::pair<double, double>
    PhaseFieldSplitSolve<dim>::calculate_total_strain_energy_and_crack_energy_dissipation() const
  {
    double total_strain_energy = 0.0;
    double crack_energy_dissipation = 0.0;

    FEValues<dim> fe_values(m_fe_phasefield, m_qf_cell, update_JxW_values);

    for (const auto &cell : m_dof_handler_phasefield.active_cell_iterators())
      {
        fe_values.reinit(cell);

        const std::vector<std::shared_ptr<const PointHistory<dim>>> lqph =
          m_quadrature_point_history.get_data(cell);
        Assert(lqph.size() == m_n_q_points, ExcInternalError());

        for (unsigned int q_point = 0; q_point < m_n_q_points; ++q_point)
          {
            const double JxW = fe_values.JxW(q_point);
            total_strain_energy += lqph[q_point]->get_total_strain_energy() * JxW;
            crack_energy_dissipation += lqph[q_point]->get_crack_energy_dissipation() * JxW;
          }
      }

    return std::make_pair(total_strain_energy, crack_energy_dissipation);
  }

  template <int dim>
  void PhaseFieldSplitSolve<dim>::print_parameter_information()
  {
    m_logfile << "Scenario number = " << m_parameters.m_scenario << std::endl;
    m_logfile << "Log file = " << m_parameters.m_logfile_name << std::endl;
    m_logfile << "Write iteration history to log file? = " << std::boolalpha
	      << m_parameters.m_output_iteration_history << std::endl;
    m_logfile << "Alternate minimization convergence criterion = "
	      << m_parameters.m_am_convergence_criterion << std::endl;
    m_logfile << "Linear solver type = " << m_parameters.m_type_linear_solver << std::endl;
    m_logfile << "Mesh refinement strategy = " << m_parameters.m_refinement_strategy << std::endl;

    if (m_parameters.m_refinement_strategy == "adaptive-refine")
      {
	m_logfile << "\tMaximum adaptive refinement times allowed in each step = "
	          << m_parameters.m_max_adaptive_refine_times << std::endl;
	m_logfile << "\tMaximum allowed cell refinement level = "
		  << m_parameters.m_max_allowed_refinement_level << std::endl;
	m_logfile << "\tPhasefield-based refinement threshold value = "
		  << m_parameters.m_phasefield_refine_threshold << std::endl;
      }

    m_logfile << "Global refinement times = " << m_parameters.m_global_refine_times << std::endl;
    m_logfile << "Local pre-refinement times = " <<m_parameters. m_local_prerefine_times << std::endl;
    m_logfile << "Allowed maximum h/l ratio = " << m_parameters.m_allowed_max_h_l_ratio << std::endl;
    m_logfile << "Total number of material types = " << m_parameters.m_total_material_regions << std::endl;
    m_logfile << "Material data file name = " << m_parameters.m_material_file_name << std::endl;
    if (m_parameters.m_reaction_force_face_id >= 0)
      m_logfile << "Calculate reaction forces on Face ID = " << m_parameters.m_reaction_force_face_id << std::endl;
    else
      m_logfile << "No need to calculate reaction forces." << std::endl;

    m_logfile << "Body force = (" << m_parameters.m_x_component << ", "
                                  << m_parameters.m_y_component << ", "
	                          << m_parameters.m_z_component << ") (N/m^3)"
				  << std::endl;

    m_logfile << "End time = " << m_parameters.m_end_time << std::endl;
    m_logfile << "Time data file name = " << m_parameters.m_time_file_name << std::endl;
  }

  template <int dim>
  bool PhaseFieldSplitSolve<dim>::local_refine_and_solution_transfer()
  {
    bool mesh_is_same = true;
    bool cell_refine_flag = true;

    unsigned int material_id;
    double length_scale;
    double cell_length;
    while(cell_refine_flag)
      {
	cell_refine_flag = false;

	std::vector<types::global_dof_index> local_dof_indices(m_fe_phasefield.dofs_per_cell);
	for (const auto &cell : m_dof_handler_phasefield.active_cell_iterators())
	  {
	    cell->get_dof_indices(local_dof_indices);

	    for (unsigned int i = 0; i< m_fe_phasefield.dofs_per_cell; ++i)
	      {
		if (  m_solution_phasefield(local_dof_indices[i])
		    > m_parameters.m_phasefield_refine_threshold )
		  {
		    material_id = cell->material_id();
		    length_scale = m_material_data[material_id][2];
		    if (dim == 2)
		      cell_length = std::sqrt(cell->measure());
		    else
		      cell_length = std::cbrt(cell->measure());
		    if (  cell_length
			> length_scale * m_parameters.m_allowed_max_h_l_ratio )
		      {
			if (cell->level() < m_parameters.m_max_allowed_refinement_level)
			  {
			    cell->set_refine_flag();
			    break;
			  }
		      }
		  }
	      }
	  }

	for (const auto &cell : m_dof_handler_phasefield.active_cell_iterators())
	  {
	    if (cell->refine_flag_set())
	      {
		cell_refine_flag = true;
		break;
	      }
	  }

	// if any cell is refined, we need to project the solution
	// to the newly refined mesh
	if (cell_refine_flag)
	  {
	    mesh_is_same = false;

	    std::vector<Vector<double> > old_solutions_d(2);
	    old_solutions_d[0] = m_solution_phasefield;
	    old_solutions_d[1] = m_solution_previous_timestep_phasefield;

	    // history variable field L2 projection
	    DoFHandler<dim> dof_handler_L2(m_triangulation);
	    FE_DGQ<dim>     fe_L2(m_parameters.m_poly_degree); //Discontinuous Galerkin
	    dof_handler_L2.distribute_dofs(fe_L2);
	    AffineConstraints<double> constraints;
	    constraints.clear();
	    DoFTools::make_hanging_node_constraints(dof_handler_L2, constraints);
	    constraints.close();

	    Vector<double> old_history_variable_field_L2;
	    old_history_variable_field_L2.reinit(dof_handler_L2.n_dofs());

	    MappingQ<dim> mapping(m_parameters.m_poly_degree + 1);
	    VectorTools::project(mapping,
				 dof_handler_L2,
				 constraints,
				 m_qf_cell,
				 [&] (const typename DoFHandler<dim>::active_cell_iterator & cell,
				      const unsigned int q) -> double
				 {
				   return m_quadrature_point_history.get_data(cell)[q]->get_history_max_positive_strain_energy();
				 },
				 old_history_variable_field_L2);

	    m_triangulation.prepare_coarsening_and_refinement();
	    SolutionTransfer<dim, Vector<double>> solution_transfer_u(m_dof_handler_displacement);
	    solution_transfer_u.prepare_for_coarsening_and_refinement(m_solution_previous_timestep_displacement);
	    SolutionTransfer<dim, Vector<double>> solution_transfer_d(m_dof_handler_phasefield);
	    solution_transfer_d.prepare_for_coarsening_and_refinement(old_solutions_d);
	    SolutionTransfer<dim, Vector<double>> solution_transfer_history_variable(dof_handler_L2);
	    solution_transfer_history_variable.prepare_for_coarsening_and_refinement(old_history_variable_field_L2);
	    m_triangulation.execute_coarsening_and_refinement();

	    setup_system();

	    dof_handler_L2.distribute_dofs(fe_L2);
	    constraints.clear();
	    DoFTools::make_hanging_node_constraints(dof_handler_L2, constraints);
	    constraints.close();

	    Vector<double> tmp_solution_previous_u;
	    tmp_solution_previous_u.reinit(m_dof_handler_displacement.n_dofs());
	    std::vector<Vector<double>> tmp_solutions_d(2);
	    tmp_solutions_d[0].reinit(m_dof_handler_phasefield.n_dofs());
	    tmp_solutions_d[1].reinit(m_dof_handler_phasefield.n_dofs());

	    Vector<double> new_history_variable_field_L2;
	    new_history_variable_field_L2.reinit(dof_handler_L2.n_dofs());

#  if DEAL_II_VERSION_GTE(9, 7, 0)
	    solution_transfer_u.interpolate(tmp_solution_previous_u);
#  else
	    // If an older version of dealII is used, for example, 9.4.0, interpolate()
            // needs to use the following interface.
	    solution_transfer_u.interpolate(m_solution_previous_timestep_displacement, tmp_solution_previous_u);
#  endif

#  if DEAL_II_VERSION_GTE(9, 7, 0)
	    solution_transfer_d.interpolate(tmp_solutions_d);
#  else
	    // If an older version of dealII is used, for example, 9.4.0, interpolate()
            // needs to use the following interface.
	    solution_transfer_d.interpolate(old_solutions_d, tmp_solutions_d);
#  endif

#  if DEAL_II_VERSION_GTE(9, 7, 0)
            solution_transfer_history_variable.interpolate(new_history_variable_field_L2);
#  else
	    // If an older version of dealII is used, for example, 9.4.0, interpolate()
            // needs to use the following interface.
            solution_transfer_history_variable.interpolate(old_history_variable_field_L2, new_history_variable_field_L2);
#  endif

	    m_solution_previous_timestep_displacement = tmp_solution_previous_u;

	    m_solution_previous_timestep_phasefield = tmp_solutions_d[1];
	    m_solution_phasefield = tmp_solutions_d[0];

	    // make sure the projected solutions still satisfy
	    // hanging node constraints
	    m_constraints_displacement.distribute(m_solution_previous_timestep_displacement);
	    m_constraints_phasefield.distribute(m_solution_previous_timestep_phasefield);
	    m_constraints_phasefield.distribute(m_solution_phasefield);
	    constraints.distribute(new_history_variable_field_L2);

	    // new_history_variable_field_L2 contains the history variable projected
	    // onto the newly refined mesh
	    FEValues<dim> fe_values(fe_L2,
				    m_qf_cell,
				    update_values | update_gradients |
				    update_quadrature_points | update_JxW_values);

	    for (const auto &cell : dof_handler_L2.active_cell_iterators())
	      {
		fe_values.reinit(cell);

		const std::vector<std::shared_ptr<PointHistory<dim>>> lqph =
		      m_quadrature_point_history.get_data(cell);

		std::vector<double> history_variable_values_cell(m_n_q_points);

		fe_values.get_function_values(
		    new_history_variable_field_L2, history_variable_values_cell);

		for (unsigned int q_point : fe_values.quadrature_point_indices())
		  {
		    lqph[q_point]->assign_history_variable(history_variable_values_cell[q_point]);
		  }
	      }
	  } // if (cell_refine_flag)
      } // while(cell_refine_flag)

    // calculate field variables for newly refined cells
    if (!mesh_is_same)
      {
	m_solution_displacement = m_solution_previous_timestep_displacement;
	m_solution_phasefield = m_solution_previous_timestep_phasefield;

	Vector<double> solution_delta_u(m_dof_handler_displacement.n_dofs());
	solution_delta_u = 0.0;
	update_qph_incremental(solution_delta_u);
	update_history_field_step();
      }

    return mesh_is_same;
  }

  template <int dim>
  void PhaseFieldSplitSolve<dim>::run()
  {
    print_parameter_information();

    read_material_data(m_parameters.m_material_file_name,
    		       m_parameters.m_total_material_regions);

    std::vector<std::array<double, 4>> time_table;

    read_time_data(m_parameters.m_time_file_name, time_table);

    make_grid();
    setup_system();
    output_results();

    while(m_time.current() < m_time.end() - m_time.get_delta_t()*1.0e-6)
      {
	m_time.increment(time_table);

	m_logfile << std::endl
		  << "Timestep " << m_time.get_timestep() << " @ " << m_time.current()
		  << 's' << std::endl;

        bool mesh_is_same = false;

	// solution from the previous time step
        m_solution_previous_timestep_displacement = m_solution_displacement;
	m_solution_previous_timestep_phasefield = m_solution_phasefield;

	double energy_functional_0 = 0.0;
	double energy_functional_previous = 0.0;
	double energy_functional_current = 0.0;
	double angle_beta = 90.0;

        // local adaptive mesh refinement loop
	unsigned int adp_refine_iteration = 0;
        for (; adp_refine_iteration < m_parameters.m_max_adaptive_refine_times + 1; ++adp_refine_iteration)
          {
            if (m_parameters.m_refinement_strategy == "adaptive-refine")
              m_logfile << "\tAdaptive refinement-"
            	        << adp_refine_iteration << ": " << std::endl;

            Vector<double> solution_phasefield_prev_iter(m_dof_handler_phasefield.n_dofs());
	    Vector<double> solution_phasefield_diff(m_dof_handler_phasefield.n_dofs());
	    Vector<double> solution_displacement_prev_iter(m_dof_handler_displacement.n_dofs());
	    Vector<double> solution_displacement_diff(m_dof_handler_displacement.n_dofs());

	    if (m_parameters.m_output_iteration_history)
	      print_conv_header();

	    unsigned int linear_solve_needed = 0;

	    unsigned int iter_am = 1;

	    for (; iter_am <= m_parameters.m_max_am_iteration; iter_am++)
	      {
		if (m_parameters.m_output_iteration_history)
		  m_logfile << '\t' << std::setw(4) << iter_am
			    << std::flush;

		// alternate minimization:
		// first, solve the displacement subproblem
		// then, solve the phase-field subproblem
		if (m_parameters.m_output_iteration_history)
		  m_logfile << "  \tDISP-sub" << std::flush;
		linear_solve_needed += displacement_step(iter_am);

		if (m_parameters.m_output_iteration_history)
		  m_logfile << "  PF-sub (linear)" << std::flush;
		phasefield_step();
		linear_solve_needed += 1;

		// calculate the displacement residual
		assemble_rhs_displacement();

		// calculate the phase-field residual
		assemble_rhs_phasefield();

		// calculate the phasefield increment
		solution_phasefield_diff = m_solution_phasefield - solution_phasefield_prev_iter;

		// calculate the displacement increment
		solution_displacement_diff = m_solution_displacement - solution_displacement_prev_iter;

		for (unsigned int i = 0; i < m_dof_handler_phasefield.n_dofs(); ++i)
		  {
		    if (m_constraints_phasefield.is_constrained(i))
		      {
			solution_phasefield_diff(i) = 0.0;
			m_system_rhs_phasefield(i) = 0.0;
		      }
		  }
		double phasefield_inc_l2 = solution_phasefield_diff.l2_norm();
		double phasefield_residual_l2 = m_system_rhs_phasefield.l2_norm();

		for (unsigned int i = 0; i < m_dof_handler_displacement.n_dofs(); ++i)
		  {
		    if (m_constraints_displacement.is_constrained(i))
		      {
			solution_displacement_diff(i) = 0.0;
			m_system_rhs_displacement(i) = 0.0;
		      }
		  }
		double displacement_inc_l2 = solution_displacement_diff.l2_norm();
		double displacement_residual_l2 = m_system_rhs_displacement.l2_norm();

		energy_functional_previous = energy_functional_current;
		energy_functional_current = calculate_energy_functional();

		if (m_parameters.m_output_iteration_history)
		  {
		    m_logfile << "  "
			      << displacement_residual_l2 << "  "
			      << phasefield_residual_l2 << "  "
			      << displacement_inc_l2 << "  "
			      << phasefield_inc_l2 << "  "
			      << std::fixed << std::setprecision(10) << std::scientific
			      << energy_functional_current
			      << std::endl;
		  }

		if (m_parameters.m_am_convergence_criterion == "SinglePass")
		  {
		    if (m_parameters.m_output_iteration_history)
		      {
			m_logfile << '\t';
			for (unsigned int i = 0; i < 133; ++i)
			  m_logfile << '_';
			m_logfile << std::endl;
		      }
		    m_logfile << "\tSingle pass for alternate minimization." << std::endl;
		    break;
		  }
		else if (m_parameters.m_am_convergence_criterion == "Energy")
		  {
		    if (iter_am == 1)
		      energy_functional_0 = energy_functional_current;

		    if (iter_am > 1)
		      angle_beta = std::atan( (energy_functional_previous - energy_functional_current) /
					      (energy_functional_0 - energy_functional_current) * (iter_am-1) )
					   * 45.0 / std::atan(1.0);

		    solution_phasefield_diff = m_solution_phasefield - solution_phasefield_prev_iter;
		    double phasefield_inc_l2 = solution_phasefield_diff.l2_norm();
		    if (m_parameters.m_output_iteration_history)
		      {
			m_logfile << std::endl;
			m_logfile << "\t\t|| d^{k+1} - d^{k} ||_2 =  " << phasefield_inc_l2 << std::endl;
			m_logfile << "\t\tEnergy functional (J) = " << energy_functional_current << std::endl;
			m_logfile << "\t\tEnergy functional tangent angle (degree) = " << angle_beta << "\n" << std::endl;
		      }

		    if ( (iter_am > 1) && (angle_beta < 1.0) && (angle_beta > 0.0) )
		      {
			if (m_parameters.m_output_iteration_history)
			  {
			    m_logfile << '\t';
			    for (unsigned int i = 0; i < 133; ++i)
			      m_logfile << '_';
			    m_logfile << std::endl;
			  }
			m_logfile << "\tAlternate minimization converges after "
				  << iter_am << " iterations based on the "
				  << m_parameters.m_am_convergence_criterion
				  << " convergence criterion."
				  << std::endl;

			m_logfile << "\tTotally " << linear_solve_needed
				  << " linear solves are required."
				  << std::endl;
			break;
		      }
		    else
		      {
			solution_phasefield_prev_iter = m_solution_phasefield;
			solution_displacement_prev_iter = m_solution_displacement;
		      }
		  }
		else if (m_parameters.m_am_convergence_criterion == "Residual")
		  {
		    if (   (iter_am > 1)
			&& (displacement_residual_l2 < m_parameters.m_tol_u_residual)
			&& (phasefield_residual_l2 < m_parameters.m_tol_d_residual)
			&& (displacement_inc_l2 < m_parameters.m_tol_u_incr)
			&& (phasefield_inc_l2 < m_parameters.m_tol_d_incr)   )
		      {
			if (m_parameters.m_output_iteration_history)
			  {
			    m_logfile << '\t';
			    for (unsigned int i = 0; i < 133; ++i)
			      m_logfile << '_';
			    m_logfile << std::endl;
			  }

			m_logfile << "\tAlternate minimization converges after "
				  << iter_am << " iterations based on the "
				  << m_parameters.m_am_convergence_criterion
				  << " convergence criterion."
				  << std::endl;

			m_logfile << "\tTotally " << linear_solve_needed
				  << " linear solves are required."
				  << std::endl;

			m_logfile << "\t\tAbsolute residual of disp. equation: "
				  << displacement_residual_l2 << std::endl;

			m_logfile << "\t\tAbsolute residual of phasefield equation: "
				  << phasefield_residual_l2 << std::endl;

			m_logfile << "\t\tAbsolute increment of disp.: "
				  << displacement_inc_l2 << std::endl;

			m_logfile << "\t\tAbsolute increment of phasefield: "
				  << phasefield_inc_l2 << std::endl;

			break;
		      }
		    else
		      {
			solution_phasefield_prev_iter = m_solution_phasefield;
			solution_displacement_prev_iter = m_solution_displacement;
		      }
		  }
		else
		  {
		    AssertThrow(false,
				ExcMessage("Selected alternate minimization convergence"
					   " strategy not implemented!"));
		  }
	      } // 	for (; iter_am <= m_parameters.m_max_am_iteration; iter_am++)

	    if (iter_am == m_parameters.m_max_am_iteration)
	      {
		m_logfile << "After " << m_parameters.m_max_am_iteration << " iterations, "
			  << "no convergence is achieved in alternate minimization "
			  << "based on the " << m_parameters.m_am_convergence_criterion
			  << "convergence criterion" << std::endl;
		AssertThrow(false,
			    ExcMessage("No convergence achieved in alternate minimization!"));
	      }

	    if (m_parameters.m_refinement_strategy == "adaptive-refine")
	      {

		if (adp_refine_iteration == m_parameters.m_max_adaptive_refine_times)
		  break;

		mesh_is_same = local_refine_and_solution_transfer();

		if (mesh_is_same)
		  break;
	      }
	    else if (m_parameters.m_refinement_strategy == "pre-refine")
	      {
	        break;
	      }
	    else
	      {
		AssertThrow(false,
		            ExcMessage("Selected mesh refinement strategy not implemented!"));
	      }
          } // ++adp_refine_iteration

	m_logfile << "\t\tUpdate history variable" << std::endl;
	update_history_field_step();

	//if (m_time.get_timestep() % 10 == 0)
	output_results();

	m_logfile << "\t\tEnergy functional (J) = " << std::fixed << std::setprecision(10) << std::scientific
	          << energy_functional_current << std::endl;

	std::pair<double, double> energy_pair = calculate_total_strain_energy_and_crack_energy_dissipation();
	m_logfile << "\t\tTotal strain energy (J) = " << std::fixed << std::setprecision(10) << std::scientific
		  << energy_pair.first << std::endl;
	m_logfile << "\t\tCrack energy dissipation (J) = " << std::fixed << std::setprecision(10) << std::scientific
		  << energy_pair.second << std::endl;

	std::pair<double, std::array<double, 3>> time_energy;
	time_energy.first = m_time.current();
	time_energy.second[0] = energy_pair.first;
	time_energy.second[1] = energy_pair.second;
	time_energy.second[2] = energy_pair.first + energy_pair.second;
	m_history_energy.push_back(time_energy);

	int face_ID = m_parameters.m_reaction_force_face_id;
	if (face_ID >= 0)
	  calculate_reaction_force(face_ID);

	write_history_data();
      } //     while(m_time.current() < m_time.end() - m_time.get_delta_t()*1.0e-6)
  }
} // namespace PhaseField


int main(int argc, char* argv[])
{
  using namespace dealii;

  if (argc != 2)
    AssertThrow(false,
    		ExcMessage("The number of arguments provided to the program has to be 2!"));

  const unsigned int dim = std::stoi(argv[1]);

  if (dim == 2 )
    {
      PhaseField::PhaseFieldSplitSolve<2> Staggered_2D("parameters.prm");
      Staggered_2D.run();
    }
  else if (dim == 3)
    {
      PhaseField::PhaseFieldSplitSolve<3> Staggered_3D("parameters.prm");
      Staggered_3D.run();
    }
  else
    {
      AssertThrow(false,
                  ExcMessage("Dimension has to be either 2 or 3"));
    }

  return 0;
}
