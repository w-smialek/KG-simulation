# distutils: language = c++
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False, language_level=3

from cpython cimport array
import array
from libc.math cimport sqrt, M_PI
# import numpy as np
  
from mycyrk.cy.cysolverNew cimport cysolve_ivp, DiffeqFuncType, WrapCySolverResult, CySolveOutput, PreEvalFunc, RK45_METHOD_INT, RK23_METHOD_INT

# cdef double L = 40.0              # length of 1-sphere in x and y in multiples of hbar/(m_e c)
# cdef int N2 = 50                  # max positive/negative mode in px and py   # max 41
# cdef int Ntot = 2*N2+1            # Total number of modes in one dimension

# def phi_to_psi_interp_cy(phi, px_interp, py_interp, factor, n2, ll):
#     psi = np.zeros(phi.shape).astype(complex)
#     for iy in range(-n2*factor,n2*factor):      # ROWS ARE Y,       FROM ROW     0 -> iy = -N2 -> Y = -L
#         for jx in range(-n2*factor,n2*factor):  # COLLUNMS ARE X,   FROM COLLUMN 0 -> jx = -N2 -> X = -L
#             for d in range(2):                  # DEPTH IS l,       d = 0 -> l = 1;  d = 1 -> q = -1
#                 row = n2*factor + iy 
#                 col = n2*factor + jx
#                 xx = jx/factor/n2*ll
#                 yy = iy/factor/n2*ll

#                 psi_xy = 0
#                 for a in range(-n2*factor,n2*factor):
#                     for b in range(-n2*factor,n2*factor):
#                         psi_xy += phi[d,a,b] * (cos(px_interp[a,b]*xx + py_interp[a,b]*yy) + 1j* sin(px_interp[a,b]*xx + py_interp[a,b]*yy))
#                 psi[d,row,col] = psi_xy
#         print(iy)
#     varphi, idtvarphi = 1/sqrt(2)*(psi[0,:,:]+psi[1,:,:]), 1/sqrt(2)*(psi[0,:,:]-psi[1,:,:])
#     return varphi, idtvarphi


cdef double ene(double px, double py, double m) noexcept nogil:
    return sqrt(m*m + px*px + py*py)

cdef void cython_diffeq(double* dy, double t, double* y, const void* args, PreEvalFunc pre_eval_func) noexcept nogil:
    
    cdef double* args_as_dbls = <double*>args

    cdef int n2 = <int>args_as_dbls[0]
    cdef double L = args_as_dbls[1]
    cdef double mass = args_as_dbls[2]
    cdef int ntot = 2*n2

    cdef double px = 0
    cdef double py = 0

    cdef int ix = 0
    cdef int iy = 0
    cdef int l  = 0
    cdef double xx = 0.
    # nx * (Ntot x 2 x 2) + ny * (2 x 2) + l * 2 + c
    while ix < ntot:
        iy = 0
        while iy < ntot:
            dbl_nx = <double>(ix - n2)
            dbl_ny = <double>(iy - n2)
            px = 2*M_PI/L*dbl_nx
            py = 2*M_PI/L*dbl_ny

            dy[ix*ntot*2*2 + iy*2*2 + 0*2 + 0] =  ene(px,py,mass) *   1.  * y[ix*ntot*2*2 + iy*2*2 + 0*2 + 1]
            dy[ix*ntot*2*2 + iy*2*2 + 0*2 + 1] = -ene(px,py,mass) *   1.  * y[ix*ntot*2*2 + iy*2*2 + 0*2 + 0]
            dy[ix*ntot*2*2 + iy*2*2 + 1*2 + 0] =  ene(px,py,mass) * (-1.) * y[ix*ntot*2*2 + iy*2*2 + 1*2 + 1]
            dy[ix*ntot*2*2 + iy*2*2 + 1*2 + 1] = -ene(px,py,mass) * (-1.) * y[ix*ntot*2*2 + iy*2*2 + 1*2 + 0]

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
        
def solver(tuple t_span, double[:] y0, double[:] coef, double[:] timesteps):
            
    cdef DiffeqFuncType diffeq = cython_diffeq
        
    cdef double* y0_ptr       = &y0[0]
    cdef unsigned int num_y   = len(y0)
    cdef double dbl_num_y     = len(y0)
    cdef int n_tsteps         = len(timesteps)

    cdef double[2] t_span_arr = [t_span[0], t_span[1]]
    cdef double* t_span_ptr   = &t_span_arr[0]

    #    MAX NUMBER OF T_EVAL TIMESTEPS: 300
    cdef double[300] c_timesteps

    cdef int i = 0
    while i < n_tsteps:
        c_timesteps[i] = timesteps[i]
        i += 1
     
    cdef double* timesteps_ptr = &c_timesteps[0]

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
        rtol = 1.0e-6,
        atol = 1.0e-7,
        args_ptr = args_ptr,
        num_extra = 0,
        max_num_steps = 0,
        max_ram_MB = 3000,
        dense_output = False,
        t_eval = NULL,#timesteps_ptr,
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