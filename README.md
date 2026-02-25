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

### Recent Updates
1. Add the adaptive mesh refinement option, remove some unused items in the parameters.prm file (Feb. 14th, 2026).
2. Add the plane-stress option to the code and input files (Feb. 15th, 2026).
3. Add the option of AT-2 (default), AT-1, and phase-field cohesive zone model (PFCZM). In the parameters.prm file, use the flag "Phase-field model type" to choose the phase-field model. Also, in the materialDataFile, extra material data entries need to be provided for the PFCZM model (Feb. 24th, 2026).

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
Jin T, Li Z, Chen K. A novel phase-field monolithic scheme for brittle crack propagation based on the limited-memory BFGS method with adaptive mesh refinement. Int J Numer Methods Eng. 2024;e7572. doi: 10.1002/nme.7572
```
@Article{2024:jin.li.ea:novel,
  author  = {Jin, Tao and Li, Zhao and Chen, Kuiying},
  title   = {A novel phase-field monolithic scheme for brittle crack propagation based on the limited-memory BFGS method with adaptive mesh refinement},
  journal = {International Journal for Numerical Methods in Engineering},
  year    = 2024,
  volume  = 125,
  number  = 22,
  pages   = {e7572},
  month   = nov,
  issn    = {1097-0207},
  url     = {https://onlinelibrary.wiley.com/doi/10.1002/nme.7572},
  doi     = {10.1002/nme.7572},
  publisher = {Wiley}
}
```
