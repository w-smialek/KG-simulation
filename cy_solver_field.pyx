# distutils: language = c++
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False, language_level=3

from cpython cimport array
import array
from libc.math cimport sqrt, M_PI, fabs, sqrt
from libc.stdio cimport printf
from libc.stdlib cimport exit
  
from mycyrk.cy.cysolverNew cimport cysolve_ivp, DiffeqFuncType, WrapCySolverResult, CySolveOutput, PreEvalFunc, RK45_METHOD_INT, RK23_METHOD_INT

cdef double m = 0.2               # mass in multiples of m_e

cdef double L = 200.0              # length of 1-sphere in x and y in multiples of hbar/(m_e c)
# cdef int N2 = 50                  # max positive/negative mode in px and py   # max 41
# cdef int Ntot = 2*N2+1            # Total number of modes in one dimension

###
      
cdef double frac1(int nx, int ny, int potnx, int potny, int ntot)noexcept nogil:
    cdef double px = 2*M_PI/L*nx
    cdef double pxshift = 2*M_PI/L*((nx - potnx)%ntot)
    cdef double py = 2*M_PI/L*ny
    cdef double pyshift = 2*M_PI/L*((ny - potny)%ntot)
    cdef double ene_val = ene(px,py)
    cdef double enemp0_val = ene(pxshift,pyshift)

    return (ene_val + enemp0_val)/(2 * sqrt(ene_val*enemp0_val))

cdef double frac2(int nx, int ny, int potnx, int potny, int ntot)noexcept nogil:
    cdef double px = 2*M_PI/L*nx
    cdef double pxshift = 2*M_PI/L*((nx - potnx)%ntot)
    cdef double py = 2*M_PI/L*ny
    cdef double pyshift = 2*M_PI/L*((ny - potny)%ntot)
    cdef double ene_val = ene(px,py)
    cdef double enemp0_val = ene(pxshift,pyshift)

    return (ene_val - enemp0_val)/(2 * sqrt(ene_val*enemp0_val))

cdef double fracv(int nx, int ny, int potnx, int potny, int ntot, double vpot)noexcept nogil:
    cdef double px = 2*M_PI/L*nx
    cdef double pxshift = 2*M_PI/L*((nx - potnx)%ntot)
    cdef double py = 2*M_PI/L*ny
    cdef double pyshift = 2*M_PI/L*((ny - potny)%ntot)
    cdef double ene_val = ene(px,py)
    cdef double enemp0_val = ene(pxshift,pyshift)
    cdef double pxshift2 = 2*M_PI/L*((nx + potnx)%ntot)
    cdef double pyshift2 = 2*M_PI/L*((ny + potny)%ntot)
    cdef double enemp0_val2 = ene(pxshift2,pyshift2)

    cdef double retval = 1/(2 * sqrt(ene_val*enemp0_val)) * (-pxshift*vpot + vpot*vpot)
    retval += 1/(2 * sqrt(ene_val*enemp0_val2)) * (-pxshift2*vpot + vpot*vpot)

    return retval


cdef double ene(double px, double py) noexcept nogil:
    return sqrt(m*m + px*px + py*py)

cdef void cython_diffeq(double* dy, double t, double* y, const void* args, PreEvalFunc pre_eval_func) noexcept nogil:
    
    cdef double* args_as_dbls = <double*>args

    cdef int n2 = <int>args_as_dbls[0]

    ###
    # For the beginning, our potential will be a plane wave with modes nx, ny
    # Then 

    cdef double pot = <double>args_as_dbls[1]
    cdef int pot_nx = <int>args_as_dbls[2]
    cdef int pot_ny = <int>args_as_dbls[3]

    # Vector potential for now is also a single mode in direction x

    cdef double vpot = <double>args_as_dbls[4]
    cdef int vpot_nx = <int>args_as_dbls[5]
    cdef int vpot_ny = <int>args_as_dbls[6]

    # cdef double pot = 0.1
    # cdef int pot_nx = 1
    # cdef int pot_ny = 0

    # # Vector potential for now is also a single mode in direction x

    # cdef double vpot = 0.0
    # cdef int vpot_nx = 0
    # cdef int vpot_ny = 0

    # printf("%i \n", n2)
    # printf("%f %f \n", pot, vpot)
    # printf("%i %i %i %i \n", pot_nx, pot_ny, vpot_nx, vpot_ny)

    # exit(0)

    ###

    cdef int ntot = 2*n2

    cdef double px = 0
    cdef double py = 0

    cdef int ix = 0
    cdef int iy = 0
    cdef double ene_val = 0.

    cdef int nx = 0
    cdef int ny = 0

    cdef double frac1_val = 0.
    cdef double frac2_val = 0.
    cdef double frac1_val2 = 0.
    cdef double frac2_val2 = 0.

    cdef double fracv_val = 0.

    cdef int point = 0
    cdef int point_sh = 0
    cdef int point_sh2 = 0
    
    # nx * (Ntot x 2 x 2) + ny * (2 x 2) + l * 2 + c
    while ix < ntot:
        iy = 0
        while iy < ntot:
            nx = (ix - n2)
            ny = (iy - n2)
            px = 2*M_PI/L*nx
            py = 2*M_PI/L*ny

            ene_val = ene(px,py)

            frac1_val = frac1(nx, ny, pot_nx, pot_ny, ntot)
            frac2_val = frac2(nx, ny, pot_nx, pot_ny, ntot)
            frac1_val2 = frac1(nx, ny, -pot_nx, -pot_ny, ntot)
            frac2_val2 = frac2(nx, ny, -pot_nx, -pot_ny, ntot)

            fracv_val = fracv(nx, ny, vpot_nx, vpot_ny, ntot, vpot)

            point = ix*ntot*2*2 + iy*2*2
            point_sh = ((ix-pot_nx)%ntot)*ntot*2*2 + ((iy-pot_ny)%ntot)*2*2
            point_sh2 = ((ix+pot_nx)%ntot)*ntot*2*2 + ((iy+pot_ny)%ntot)*2*2


            dy[point + 0*2 + 0] =  ene_val *   1.  * y[point + 0*2 + 1] + pot * (frac1_val * y[point_sh + 0*2 + 1] + frac2_val * y[point_sh + 1*2 + 1]) + pot * (frac1_val2 * y[point_sh2 + 0*2 + 1] + frac2_val2 * y[point_sh2 + 1*2 + 1])
            dy[point + 0*2 + 1] = -ene_val *   1.  * y[point + 0*2 + 0] - pot * (frac1_val * y[point_sh + 0*2 + 0] + frac2_val * y[point_sh + 1*2 + 0]) - pot * (frac1_val2 * y[point_sh2 + 0*2 + 0] + frac2_val2 * y[point_sh2 + 1*2 + 0])
            dy[point + 1*2 + 0] =  ene_val * (-1.) * y[point + 1*2 + 1] + pot * (frac1_val * y[point_sh + 1*2 + 1] + frac2_val * y[point_sh + 0*2 + 1]) + pot * (frac1_val2 * y[point_sh2 + 1*2 + 1] + frac2_val2 * y[point_sh2 + 0*2 + 1])
            dy[point + 1*2 + 1] = -ene_val * (-1.) * y[point + 1*2 + 0] - pot * (frac1_val * y[point_sh + 1*2 + 0] + frac2_val * y[point_sh + 0*2 + 0]) - pot * (frac1_val2 * y[point_sh2 + 1*2 + 0] + frac2_val2 * y[point_sh2 + 0*2 + 0])

            # vpot

            dy[point + 0*2 + 0] += fracv_val*(y[point + 0*2 + 1] + y[point + 1*2 + 1])
            dy[point + 0*2 + 1] -= fracv_val*(y[point + 0*2 + 0] + y[point + 1*2 + 0])
            dy[point + 1*2 + 0] += fracv_val*(y[point + 0*2 + 1] + y[point + 1*2 + 1])
            dy[point + 1*2 + 1] -= fracv_val*(y[point + 0*2 + 0] + y[point + 1*2 + 0])

            iy += 1
        ix += 1
        
def solver(tuple t_span, double[:] y0, double[:] coef, double[:] timesteps):
            
    cdef DiffeqFuncType diffeq = cython_diffeq
        
    cdef double* y0_ptr       = &y0[0]
    cdef unsigned int num_y   = len(y0)
    cdef double dbl_num_y     = len(y0)
    cdef int n_tsteps         = len(timesteps)

    cdef double[2] t_span_arr = [t_span[0], t_span[1]]
    cdef double* t_span_ptr   = &t_span_arr[0]

    #    MAX NUMBER OF T_EVAL TIMESTEPS: 3000
    cdef double[3000] c_timesteps

    cdef int i = 0
    while i < n_tsteps:
        c_timesteps[i] = timesteps[i]
        i += 1
    
    cdef double* timesteps_ptr = &c_timesteps[0]

    # Assume constant args
    cdef double[7] args = [coef[0], coef[1], coef[2], coef[3], coef[4], coef[5], coef[6]]
    cdef double* args_dbl_ptr = &args[0]
    # Need to cast the arg double pointer to void
    cdef void* args_ptr = <void*>args_dbl_ptr
   
    cdef CySolveOutput result = cysolve_ivp(
        diffeq,
        t_span_ptr,
        y0_ptr,
        num_y,
        method = RK45_METHOD_INT,
        rtol = 1.0e-5,
        atol = 1.0e-6,
        args_ptr = args_ptr,
        num_extra = 0,
        max_num_steps = 0,
        max_ram_MB = 3000,
        dense_output = False,
        t_eval = timesteps_ptr,
        len_t_eval = n_tsteps, 
        pre_eval_func = NULL,
        rtols_ptr = NULL,
        atols_ptr = NULL,
        first_step = 0.0,
        expected_size = 0,
        max_step = 10000000000.0
    )
 
    cdef WrapCySolverResult pysafe_result = WrapCySolverResult()
    pysafe_result.set_cyresult_pointer(result)

    return pysafe_result