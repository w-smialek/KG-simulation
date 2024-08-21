# Numerical simulation of the complex Klein-Gordon equation

This project contains an implementation of cython-based
solver for the time evolution of a complex Klein-Gordon field coupled to electromagnetic potential:

$$ \left( (\partial_\mu - ieA_\mu)^2 +m^2 \right) \varphi = 0 $$

At this stage, the solver handles an electric potential constant in time:
$$A_\mu (t,\mathbf{x}) = (V(\mathbf{x}),0)$$

The domain of the simulation is a 2-torus, i.e. a square with periodic boundary conditions. The advantage of such domain is that the only necessary approximation when storing the field value is introducing a momentum cutoff for a finite grid, while the discretization of momenta is exact.

The solver works in the Feshbach-Villars representation of Klein-Gordon field (cf. https://doi.org/10.1103/RevModPhys.30.24), but it provides tools for transforming between different representations.

No finite difference method is used; when working in momentum space the integration step involves just point-wise multiplications and convolutions, which should greatly improve numerical accuracy.

> **_NOTE:_**  The project is currently work in progress!

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

## Documentation

(Section to be expanded...)

Consult the example kg_field.py for basic usage

## Examples

1. Low - energy scattering off of the potential step

In the low energy regime, the scattering works similarly to non-relativistic quantum mechanics.

The following examples show simulations with mean momentum  $p_x = 0.5 mc$
and potential step $V_0 = 0.10,0.11,0.12,0.13 \ mc^2/e$,
near the transition from the bound to the free solution

 ![](./gifs/anim_pot0.10_mom0.25_d.gif)
 ![](./gifs/anim_pot0.11_mom0.25_d.gif)
 ![](./gifs/anim_pot0.12_mom0.25_d.gif)
 ![](./gifs/anim_pot0.13_mom0.25_d.gif)

1. Klein paradox: high - energy scattering off of the potential step

Above the limit $V_0 e = 2mc^2$, the transmission coefficient may become higher than 1, and develops a singularity for certain momenta. At the sharp potential step, positive and negative charges begin to be produced, the total charge however (up to the numerical accuracy) stays constant.
You can see the [interactive graph](https://www.desmos.com/calculator/w4ljbuavg9) for the transmission coefficient with different values of the potential barrier.
The Klein paradox in some way suggests the actual phenomena of the Quantum Field Theory, such as the Schwinger effect, which occur for similar range of values of the electric field.

The following examples show simulations with mean momentum  $p_x = 0.5 mc$
and potential step $V_0 = 2.10,2.13,2.90 \ mc^2/e$,
near the transition between two scattering regimes

 ![](./gifs/anim_pot2.10_mom0.25_d.gif)
 ![](./gifs/anim_pot2.13_mom0.25_d.gif)
 ![](./gifs/anim_pot2.90_mom0.25_d.gif)
