## Phasefield staggered approach
The staggered approach based on the alternate minimization for phase-field fracture simulations.
### Purpose
This repository is created mainly as a baseline for comparing its convergence behavior with various phase-field monolithic solvers, such as the limited-memory BFGS (L-BFGS) and the L-BFGS-B schemes. The staggered approach has the following features:

1. At each load (pseudo-time) step, the displacement sub-problem (nonlinear) and the phase-field sub-problem are solved alternatively.
2. Several options of convergence criteria are provided, including the single-pass criteria (not practical), the energy-based criteria, and the residual-based criteria.
3. The damage irreversibility is enforced using the history variable (maximum positive strain energy) approach.

### Content
The repository contains the following content:
1. the source code of the alternate minimization method for the phase-field staggered approach.
2. the input files for several commonly used 2D phase-field fracture simulations.

### How to compile
The L-BFGS finite element procedure is implemented in [deal.II](https://www.dealii.org/) (with the develop branch as Aug. 17th, 2024), which is an open source finite element library. In order to use the code (**main.cc**) provided here, deal.II should be configured with MPI and at least with the interfaces to BLAS, LAPACK, Threading Building Blocks (TBB), and UMFPACK. For optional interfaces to other software packages, see https://www.dealii.org/developer/readme.html.
Once the deal.II library is compiled, for instance, to "~/dealii-dev/bin/", follow the steps listed below:
1. cd SourceCode
2. cmake -DDEAL_II_DIR=~/dealii-dev/bin/  .
3. make debug or make release
4. make

### How to run
1. Go into one of the examples folders.
2. For instance a 2D test case: go into simple_shear/
3. Run via ./../SourceCode/main 2

### How to cite this work:
TBD
