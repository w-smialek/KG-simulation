# distutils: language = c++
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False, language_level=3

from cpython cimport array
import array
from libc.math cimport sqrt, M_PI, fabs
from libc.stdio cimport printf
  
from mycyrk.cy.cysolverNew cimport cysolve_ivp, DiffeqFuncType, WrapCySolverResult, CySolveOutput, PreEvalFunc, RK45_METHOD_INT

cdef double m = 1.0               # mass in multiples of m_e

cdef double L = 40.0              # length of 1-sphere in x and y in multiples of hbar/(m_e c)
cdef int N2 = 40                  # max positive/negative mode in px and py   # max 41
cdef int Ntot = 2*N2+1            # Total number of modes in one dimension



cdef double ene(double px, double py) noexcept nogil:
    return sqrt(m*m + px*px + py*py)

cdef void cython_diffeq(double* dy, double t, double* y, const void* args, PreEvalFunc pre_eval_func) noexcept nogil:
    
    cdef double* args_as_dbls = <double*>args

    cdef double a = args_as_dbls[0]
    cdef double b = args_as_dbls[1]
    cdef int num_y = <int>args_as_dbls[2]

    cdef double px = 0
    cdef double py = 0

    cdef int ix = 0
    cdef int iy = 0
    cdef int l  = 0
    cdef double xx = 0.
    # nx * (Ntot x 2 x 2) + ny * (2 x 2) + l * 2 + c
    while ix < Ntot:
        iy = 0
        while iy < Ntot:
            dbl_nx = <double>(ix - N2)
            dbl_ny = <double>(iy - N2)
            px = 2*M_PI/L*dbl_nx
            py = 2*M_PI/L*dbl_ny

            # printf("%f ",dbl_nx)
            # printf("%f ",dbl_ny)

            dy[ix*Ntot*2*2 + iy*2*2 + 0*2 + 0] =  ene(px,py) *   1.  * y[ix*Ntot*2*2 + iy*2*2 + 0*2 + 1]
            dy[ix*Ntot*2*2 + iy*2*2 + 0*2 + 1] = -ene(px,py) *   1.  * y[ix*Ntot*2*2 + iy*2*2 + 0*2 + 0]
            dy[ix*Ntot*2*2 + iy*2*2 + 1*2 + 0] =  ene(px,py) * (-1.) * y[ix*Ntot*2*2 + iy*2*2 + 1*2 + 1]
            dy[ix*Ntot*2*2 + iy*2*2 + 1*2 + 1] = -ene(px,py) * (-1.) * y[ix*Ntot*2*2 + iy*2*2 + 1*2 + 0]

            # l = 0
            # while l < 4:
            #     xx = y[ix*Ntot*2*2 + iy*2*2 + l]
            #     if fabs(xx) > 0.001:
            #         printf("%f ", dbl_nx)   
            #         printf("%f\n", dbl_ny)   
            #         printf("%f\n", ene(px,py))
            #     l += 1             

            iy += 1
        ix += 1
        
def solver(tuple t_span, double[:] y0, double[:] coef):
    print('solver')
            
    cdef DiffeqFuncType diffeq = cython_diffeq
        
    cdef double* y0_ptr       = &y0[0]
    cdef unsigned int num_y   = len(y0)
    cdef double dbl_num_y     = len(y0)

    cdef double[2] t_span_arr = [t_span[0], t_span[1]]
    cdef double* t_span_ptr   = &t_span_arr[0]

    # Assume constant args
    cdef double[3] args = [coef[0], coef[1], dbl_num_y]
    cdef double* args_dbl_ptr = &args[0]
    # Need to cast the arg double pointer to void
    cdef void* args_ptr = <void*>args_dbl_ptr

    cdef CySolveOutput result = cysolve_ivp(
        diffeq,
        t_span_ptr,
        y0_ptr,
        num_y,
        method = RK45_METHOD_INT,
        args_ptr = args_ptr,
        rtol = 1.0e-9,
        atol = 1.0e-10,
        num_extra = 0
    )
 
    cdef WrapCySolverResult pysafe_result = WrapCySolverResult()
    pysafe_result.set_cyresult_pointer(result)

    return pysafe_result