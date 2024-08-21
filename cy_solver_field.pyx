# distutils: language = c++
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False, language_level=3

from cpython cimport array
import array
from libc.math cimport sqrt, M_PI, sqrt
from libc.stdlib cimport malloc, free
  
from mycyrk.cy.cysolverNew cimport cysolve_ivp, DiffeqFuncType, WrapCySolverResult, CySolveOutput, PreEvalFunc, RK45_METHOD_INT, RK23_METHOD_INT
cimport fft_conv.src.fftlib as ft

###
      
cdef double ene(double m, double px, double py) noexcept nogil:
    return sqrt(m*m + px*px + py*py)

cdef double* create_ene1(int n2, double m, double l) noexcept nogil:
    cdef int ntot = 2*n2
    cdef int ix = 0
    cdef int iy = 0
    cdef double *ene1 = <double*>malloc(sizeof(double) * ntot*ntot)
    cdef double px
    cdef double py
    while ix < ntot:
        iy = 0
        px = 2*M_PI/l*(ix-n2)
        while iy < ntot:
                
            py = 2*M_PI/l*(iy-n2)
            ene1[ix*ntot + iy] = sqrt(ene(m,px,py))

            iy+=1
        ix+=1
    return &ene1[0]

cdef double* create_ene2(int n2, double m, double l) noexcept nogil:
    cdef int ntot = 2*n2
    cdef int ix = 0
    cdef int iy = 0
    cdef double *ene2 = <double*>malloc(sizeof(double) * ntot*ntot)
    cdef double px
    cdef double py
    while ix < ntot:
        iy = 0
        px = 2*M_PI/l*(ix-n2)
        while iy < ntot:
                
            py = 2*M_PI/l*(iy-n2)
            ene2[ix*ntot + iy] = 1/(2*sqrt(ene(m,px,py)))

            iy+=1
        ix+=1
    return &ene2[0]

###
### Parameters
###

cdef struct parameters:
    int n2
    double m
    double l
    ft.fft_data* pdata
    double complex* field
    double* ene1_ptr
    double* ene2_ptr

###
### CyRK
###

cdef void cython_diffeq(double* dy, double t, double* y, const void* args, PreEvalFunc pre_eval_func) noexcept nogil:
    
    cdef parameters* args_ptr = <parameters*>args
    cdef int n2 = args_ptr.n2
    cdef double m = args_ptr.m
    cdef double l = args_ptr.l
    cdef ft.fft_data* pdata = args_ptr.pdata
    cdef double complex* potential_c = args_ptr.field
    cdef double* ene1_ptr = args_ptr.ene1_ptr
    cdef double* ene2_ptr = args_ptr.ene2_ptr

    cdef int ntot = 2*n2
    cdef double px = 0
    cdef double py = 0
    cdef int ix = 0
    cdef int iy = 0
    cdef double ene_val = 0.
    cdef int nx = 0
    cdef int ny = 0
    cdef int point = 0
    cdef int point_ene = 0
    
    cdef double complex conv1 = 0
    cdef double complex conv2 = 0
    cdef double complex temp1c = 0
    cdef double complex temp2c = 0

    cdef double complex* kernel1 = <double complex*>malloc(ntot*ntot*sizeof(double complex))
    cdef double complex* kernel2 = <double complex*>malloc(ntot*ntot*sizeof(double complex))
    cdef double complex* result1 = <double complex*>malloc(ntot*ntot*sizeof(double complex))
    cdef double complex* result2 = <double complex*>malloc(ntot*ntot*sizeof(double complex))

    cdef int point_q = 0
    cdef int point_q_y = 0
    cdef int aix = 0
    cdef int aiy = 0
    while aix<ntot:
        aiy=0
        while aiy<ntot:
            point_q_y = (aix)*ntot*2*2 + (aiy)*2*2
            point_q = (aix)*ntot + (aiy)

            kernel1[point_q] = ene2_ptr[point_q]*(y[point_q_y] + 1j*y[point_q_y+1] + y[point_q_y+2] + 1j*y[point_q_y+3])
            kernel2[point_q] = ene1_ptr[point_q]*(y[point_q_y] + 1j*y[point_q_y+1] - y[point_q_y+2] - 1j*y[point_q_y+3])

            aiy+=1
        aix+=1

    cdef double complex* temp = <double complex*> malloc(ntot*ntot*sizeof(double complex))

    ft.convolved(pdata, result1, temp, potential_c, kernel1, n2) 
    ft.convolved(pdata, result2, temp, potential_c, kernel2, n2)

    while ix < ntot:
        iy = 0
        while iy < ntot:
            nx = (ix - n2)
            ny = (iy - n2)
            px = 2*M_PI/l*nx
            py = 2*M_PI/l*ny

            ene_val = ene(m,px,py)

            point = ix*ntot*2*2 + iy*2*2

            dy[point + 0*2 + 0] =  ene_val *   1.  * y[point + 0*2 + 1]
            dy[point + 0*2 + 1] = -ene_val *   1.  * y[point + 0*2 + 0]
            dy[point + 1*2 + 0] =  ene_val * (-1.) * y[point + 1*2 + 1]
            dy[point + 1*2 + 1] = -ene_val * (-1.) * y[point + 1*2 + 0]

            point_ene = (ix)*ntot + (iy)

            conv1 = result1[ix*ntot+iy]
            conv2 = result2[ix*ntot+iy]

            temp1c = (-1j)*(ene1_ptr[point_ene]*conv1 + ene2_ptr[point_ene]*conv2)
            temp2c = (-1j)*(ene1_ptr[point_ene]*conv1 - ene2_ptr[point_ene]*conv2)

            dy[point + 0*2 + 0] += temp1c.real
            dy[point + 0*2 + 1] += temp1c.imag
            dy[point + 1*2 + 0] += temp2c.real
            dy[point + 1*2 + 1] += temp2c.imag

            iy += 1
        ix += 1
    
    free(kernel1)
    free(kernel2)
    free(result1)
    free(result2)
    free(temp)

        
def solver(tuple t_span, double[:] y0, double[:] coef, double[:] timesteps, double[:] field):
            
    cdef DiffeqFuncType diffeq = cython_diffeq
        
    cdef double* y0_ptr       = &y0[0]
    cdef unsigned int num_y   = len(y0)
    cdef int n_tsteps         = len(timesteps)
    cdef int n2               = <int>coef[0]
    cdef int ntot             = 2*n2

    cdef double[2] t_span_arr = [t_span[0], t_span[1]]
    cdef double* t_span_ptr   = &t_span_arr[0]

    #    Timesteps
    #    MAX NUMBER OF T_EVAL TIMESTEPS: 1000

    cdef double[1000] c_timesteps
    cdef int i = 0
    cdef int j = 0
    while i < n_tsteps:
        c_timesteps[i] = timesteps[i]
        i += 1
    cdef double* timesteps_ptr = &c_timesteps[0]

    #    Prepare parameters

    cdef parameters args
    args.n2 = n2
    cdef ft.fft_data data
    ft.setup(&data, ntot, ntot)
    args.pdata = &data
    args.m = coef[1]
    args.l = coef[2]
    cdef parameters* args_param_ptr = &args

    cdef double complex* potential_c = <double complex*>malloc(ntot*ntot*sizeof(double complex))
    i = 0
    while i<ntot:
        j = 0
        while j<ntot:
            potential_c[i*ntot + j] = field[i*ntot*2 + j*2] + 1j*field[i*ntot*2 + j*2 +1]
            j+=1
        i+=1

    args.field = potential_c
    args.ene1_ptr = create_ene1(n2,coef[1],coef[2])
    args.ene2_ptr = create_ene2(n2,coef[1],coef[2])

    # Need to cast the arg double pointer to void
    cdef void* args_ptr = <void*>args_param_ptr
   
    cdef CySolveOutput result = cysolve_ivp(
        diffeq,
        t_span_ptr,
        y0_ptr,
        num_y,
        method = RK45_METHOD_INT,
        rtol = 1.0e-9,
        atol = 1.0e-9,
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