# Klein-Gordon equation simulation

This project contains an implementation of cython-based
solver for the time evolution of a complex Klein-Gordon field coupled to electromagnetic potential:

$$ \left( (\partial_\mu - ieA_\mu)^2 +m^2 \right) \varphi = 0 $$

At this stage, the solver handles an electric potential constant in time:
$$ A_\mu (t,\vec{x}) = (V(\vec{x}),0) $$

The domain of the simulation is a 2-torus, i.e. a square with periodic boundary conditions. The advantage of such domain is that the only necessary approximation when storing the field value is introducing a momentum cutoff for a finite grid, while the discretization of momenta is exact.

The solver works in the Feshbach-Villars representation of Klein-Gordon field (cf. https://doi.org/10.1103/RevModPhys.30.24), but it provides tools for transforming between different representations.

No finite difference method is used; when working in momentum space the integration step involves just point-wise multiplications and convolutions, which should greatly improve numerical accuracy.

## Solver tools

The solver is based on Runge-Kutta ODE Integrator [**CyRK**](https://github.com/jrenaud90/CyRK), with some modifications allowing for a large size of the system of ODEs, required for partial differential equation such as the Klein-Gordon equation.

CyRK library, as well as this project, are shared under Creative Commons Attribution-ShareAlike 4.0 International license

The convolution implemented in the solver is based on the Fast Fourier Transform and uses the FFTW 3.3.10 library

## Installation

The project requires installation of the following python libraries:

- Numpy
- Scipy
- Matplotlib
- PIL

Additionally, it requires C library FFTW 3 \
To install the latest version of FFTW on linux system, run
```
sudo apt-get update
sudo apt-get install libfftw3-3 
```
To build the project, one requires
python setuptools and cython library installed, as well
as a gcc compiler.
Open the main folder and run
```
python3 setup.py build_ext
```