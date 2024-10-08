# distutils: language = c++
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False, language_level=3

from cpython cimport array
import array
from libc.math cimport sqrt, M_PI, sqrt, fabs
from libc.stdlib cimport malloc, free
from libc.stdio cimport printf
  
from mycyrk.cy.cysolverNew cimport cysolve_ivp, DiffeqFuncType, WrapCySolverResult, CySolveOutput, PreEvalFunc, RK45_METHOD_INT, RK23_METHOD_INT
cimport fft_conv.src.fftlib as ft
import numpy as np
import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt


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
    double complex* field_a1
    double complex* field_a2
    double complex* field_a0
    double* ene1_ptr
    double* ene2_ptr

cdef struct parameters_t:
    int n2
    int n_t
    double m
    double l
    ft.fft_data* pdata
    double complex* field_interp
    double complex* field_a1_interp
    double complex* field_a2_interp
    double complex* field_a0_interp
    double* t_knots
    int is_field   
    int is_field_a1
    int is_field_a2
    int is_field_a0
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
    cdef double complex* potential_ca1 = args_ptr.field_a1
    cdef double complex* potential_ca2 = args_ptr.field_a2
    cdef double complex* potential_ca0 = args_ptr.field_a0
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

    # ix = 0
    # while ix < ntot:
    #     iy = 0
    #     while iy < ntot:
    #         if fabs(potential_ca2[ix*ntot+iy].imag) > 0.5:
    #             printf('%.3f %i %i\n',fabs(potential_ca2[ix*ntot+iy].imag),ix,iy)
    #         iy += 1
    #     ix += 1
    # ix = 0
    # iy = 0

    cdef double complex conv1 = 0
    cdef double complex conv2 = 0
    cdef double complex temp1c = 0
    cdef double complex temp2c = 0

    cdef double complex conv_a0 = 0
    cdef double complex conv_a1 = 0
    cdef double complex conv_a2 = 0
    cdef double complex tempa0c = 0
    cdef double complex tempa1c = 0
    cdef double complex tempa2c = 0
    cdef double complex kgfieldc = 0

    cdef double complex* kernel1 = <double complex*>malloc(ntot*ntot*sizeof(double complex))
    cdef double complex* kernel2 = <double complex*>malloc(ntot*ntot*sizeof(double complex))
    cdef double complex* result1 = <double complex*>malloc(ntot*ntot*sizeof(double complex))
    cdef double complex* result2 = <double complex*>malloc(ntot*ntot*sizeof(double complex))

    cdef double complex* kernel0_a = <double complex*>malloc(ntot*ntot*sizeof(double complex))
    cdef double complex* kernel1_a = <double complex*>malloc(ntot*ntot*sizeof(double complex))
    cdef double complex* kernel2_a = <double complex*>malloc(ntot*ntot*sizeof(double complex))
    cdef double complex* result0_a = <double complex*>malloc(ntot*ntot*sizeof(double complex))
    cdef double complex* result1_a = <double complex*>malloc(ntot*ntot*sizeof(double complex))
    cdef double complex* result2_a = <double complex*>malloc(ntot*ntot*sizeof(double complex))

    cdef int point_q = 0
    cdef int point_q_y = 0
    cdef int aix = 0
    cdef int aiy = 0
    cdef double px_aix = 0
    cdef double py_aiy = 0
    while aix<ntot:
        aiy=0
        while aiy<ntot:
            point_q_y = (aix)*ntot*2*2 + (aiy)*2*2
            point_q = (aix)*ntot + (aiy)
            px_aix = 2*M_PI/l*(aix-n2)
            py_aiy = 2*M_PI/l*(aiy-n2)

            kernel1[point_q] = ene2_ptr[point_q]*(y[point_q_y] + 1j*y[point_q_y+1] + y[point_q_y+2] + 1j*y[point_q_y+3])
            kernel2[point_q] = ene1_ptr[point_q]*(y[point_q_y] + 1j*y[point_q_y+1] - y[point_q_y+2] - 1j*y[point_q_y+3])

            kernel0_a[point_q] = ene2_ptr[point_q] * (y[point_q_y] + 1j*y[point_q_y+1] + y[point_q_y+2] + 1j*y[point_q_y+3])
            kernel1_a[point_q] = ene2_ptr[point_q] * px * (y[point_q_y] + 1j*y[point_q_y+1] + y[point_q_y+2] + 1j*y[point_q_y+3])
            kernel2_a[point_q] = ene2_ptr[point_q] * py * (y[point_q_y] + 1j*y[point_q_y+1] + y[point_q_y+2] + 1j*y[point_q_y+3])

            aiy+=1
        aix+=1

    cdef double complex* temp = <double complex*> malloc(ntot*ntot*sizeof(double complex))

    ft.convolved(pdata, result1, temp, potential_c, kernel1, n2) 
    ft.convolved(pdata, result2, temp, potential_c, kernel2, n2)
    
    ft.convolved(pdata, result1_a, temp, potential_ca1, kernel1_a, n2)
    ft.convolved(pdata, result2_a, temp, potential_ca2, kernel2_a, n2)
    ft.convolved(pdata, result0_a, temp, potential_ca0, kernel0_a, n2)

    while ix < ntot:
        iy = 0
        while iy < ntot:
            nx = (ix - n2)
            ny = (iy - n2)
            px = M_PI/l*nx
            py = M_PI/l*ny

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

            point_ene = (ix)*ntot + (iy)
            kgfieldc = (y[point + 0*2 + 0] + 1j*y[point + 0*2 + 1] + y[point + 1*2 + 0] + 1j*y[point + 0*2 + 1])

            conv_a0 = ene2_ptr[point_ene]*result0_a[ix*ntot+iy]
            conv_a1 = -2*ene2_ptr[point_ene]*result1_a[ix*ntot+iy]
            conv_a2 = -2*ene2_ptr[point_ene]*result2_a[ix*ntot+iy]

            tempa0c = (-1j)*(conv_a0+conv_a1+conv_a2)
            tempa1c = (-1j)*(conv_a0+conv_a1+conv_a2)

            dy[point + 0*2 + 0] += tempa0c.real
            dy[point + 0*2 + 1] += tempa0c.imag
            dy[point + 1*2 + 0] += tempa1c.real
            dy[point + 1*2 + 1] += tempa1c.imag

            iy += 1
        ix += 1
    
    free(kernel1)
    free(kernel2)
    free(result1)
    free(result2)
    free(kernel0_a)
    free(kernel1_a)
    free(kernel2_a)
    free(result0_a)
    free(result1_a)
    free(result2_a)
    free(temp)

        
def solver(tuple t_span, double[:] y0, double[:] coef, double[:] timesteps, 
           double[:] field, double[:] field_a1, double[:] field_a2, double[:] field_a0):

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

    cdef double complex* potential_c = <double complex*>malloc(ntot*ntot*sizeof(double complex))
    cdef double complex* potential_ca1 = <double complex*>malloc(ntot*ntot*sizeof(double complex))
    cdef double complex* potential_ca2 = <double complex*>malloc(ntot*ntot*sizeof(double complex))
    cdef double complex* potential_ca0 = <double complex*>malloc(ntot*ntot*sizeof(double complex))

    cdef ft.fft_data data
    ft.setup(&data, ntot, ntot)

    cdef parameters args
    cdef parameters* args_param_ptr = &args
    args.n2 = n2
    args.pdata = &data
    args.m = coef[1]
    args.l = coef[2]


    i = 0
    while i<ntot:
        j = 0
        while j<ntot:
            potential_c[i*ntot + j] = field[i*ntot*2 + j*2] + 1j*field[i*ntot*2 + j*2 +1]
            j+=1
        i+=1
            
    i = 0
    while i<ntot:
        j = 0
        while j<ntot:
            potential_ca1[i*ntot + j] = field_a1[i*ntot*2 + j*2] + 1j*field_a1[i*ntot*2 + j*2 +1]
            j+=1
        i+=1

    i = 0
    while i<ntot:
        j = 0
        while j<ntot:
            potential_ca2[i*ntot + j] = field_a2[i*ntot*2 + j*2] + 1j*field_a2[i*ntot*2 + j*2 +1]
            j+=1
        i+=1

    i = 0
    while i<ntot:
        j = 0
        while j<ntot:
            potential_ca0[i*ntot + j] = field_a0[i*ntot*2 + j*2] + 1j*field_a0[i*ntot*2 + j*2 +1]
            j+=1
        i+=1

    args.field = potential_c
    args.field_a1 = potential_ca1
    args.field_a2 = potential_ca2
    args.field_a0 = potential_ca0

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

###
### Time-dependent field
###

cdef void cython_diffeq_t(double* dy, double t, double* y, const void* args, PreEvalFunc pre_eval_func) noexcept nogil:
    
    cdef parameters_t* args_ptr = <parameters_t*>args
    cdef int n2 = args_ptr.n2
    cdef int n_t = args_ptr.n_t
    cdef double m = args_ptr.m
    cdef double l = args_ptr.l
    cdef double* t_knots = args_ptr.t_knots
    cdef ft.fft_data* pdata = args_ptr.pdata
    cdef double complex* potential_c_interp = args_ptr.field_interp
    cdef double complex* potential_ca1_interp = args_ptr.field_a1_interp
    cdef double complex* potential_ca2_interp = args_ptr.field_a2_interp
    cdef double complex* potential_ca0_interp = args_ptr.field_a0_interp
    cdef double* ene1_ptr = args_ptr.ene1_ptr
    cdef double* ene2_ptr = args_ptr.ene2_ptr

    cdef int i_knot = 0

    while not (t_knots[i_knot] <= t < t_knots[i_knot +1] or i_knot == n_t-2):
        i_knot += 1
    
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

    cdef double complex* potential_c = <double complex*>malloc(ntot*ntot*sizeof(double complex))
    cdef double complex* potential_ca1 = <double complex*>malloc(ntot*ntot*sizeof(double complex))
    cdef double complex* potential_ca2 = <double complex*>malloc(ntot*ntot*sizeof(double complex))
    cdef double complex* potential_ca0 = <double complex*>malloc(ntot*ntot*sizeof(double complex))

    ix = 0
    while ix < ntot:
        iy = 0
        while iy < ntot:
            potential_c[ix*ntot+iy] = 0
            potential_ca1[ix*ntot+iy] = 0
            potential_ca2[ix*ntot+iy] = 0
            potential_ca0[ix*ntot+iy] = 0
            iy += 1
        ix += 1

    cdef int p = 0
    
    ix = 0
    while ix < ntot:
        iy = 0
        while iy < ntot:
            p = 0
            while p < 4:
                potential_ca1[ix*ntot+iy] += (t-t_knots[i_knot])**(3-p)*potential_ca1_interp[p*n_t*ntot*ntot+i_knot*ntot*ntot+ix*ntot+iy]
                p += 1
            iy += 1
        ix += 1

    ix = 0
    while ix < ntot:
        iy = 0
        while iy < ntot:
            p = 0
            while p < 4:
                potential_ca2[ix*ntot+iy] += (t-t_knots[i_knot])**(3-p)*potential_ca2_interp[p*n_t*ntot*ntot+i_knot*ntot*ntot+ix*ntot+iy]
                p += 1
            # if fabs(potential_ca2[ix*ntot+iy].imag) > 0.5:
            #     printf('%.3f %i %i\n',fabs(potential_ca2[ix*ntot+iy].imag),ix,iy)
            iy += 1
        ix += 1

    ix = 0
    while ix < ntot:
        iy = 0
        while iy < ntot:
            p = 0
            while p < 4:
                potential_ca0[ix*ntot+iy] += (t-t_knots[i_knot])**(3-p)*potential_ca0_interp[p*n_t*ntot*ntot+i_knot*ntot*ntot+ix*ntot+iy]
                p += 1
            iy += 1
        ix += 1

    cdef double complex conv1 = 0
    cdef double complex conv2 = 0
    cdef double complex temp1c = 0
    cdef double complex temp2c = 0

    cdef double complex conv_a0 = 0
    cdef double complex conv_a1 = 0
    cdef double complex conv_a2 = 0
    cdef double complex tempa0c = 0
    cdef double complex tempa1c = 0
    cdef double complex tempa2c = 0
    cdef double complex kgfieldc = 0

    cdef double complex* kernel1 = <double complex*>malloc(ntot*ntot*sizeof(double complex))
    cdef double complex* kernel2 = <double complex*>malloc(ntot*ntot*sizeof(double complex))
    cdef double complex* result1 = <double complex*>malloc(ntot*ntot*sizeof(double complex))
    cdef double complex* result2 = <double complex*>malloc(ntot*ntot*sizeof(double complex))

    cdef double complex* kernel0_a = <double complex*>malloc(ntot*ntot*sizeof(double complex))
    cdef double complex* kernel1_a = <double complex*>malloc(ntot*ntot*sizeof(double complex))
    cdef double complex* kernel2_a = <double complex*>malloc(ntot*ntot*sizeof(double complex))
    cdef double complex* result0_a = <double complex*>malloc(ntot*ntot*sizeof(double complex))
    cdef double complex* result1_a = <double complex*>malloc(ntot*ntot*sizeof(double complex))
    cdef double complex* result2_a = <double complex*>malloc(ntot*ntot*sizeof(double complex))

    cdef int point_q = 0
    cdef int point_q_y = 0
    cdef int aix = 0
    cdef int aiy = 0
    cdef double px_aix = 0
    cdef double py_aiy = 0
    while aix<ntot:
        aiy=0
        while aiy<ntot:
            point_q_y = (aix)*ntot*2*2 + (aiy)*2*2
            point_q = (aix)*ntot + (aiy)
            px_aix = 2*M_PI/l*(aix-n2)
            py_aiy = 2*M_PI/l*(aiy-n2)

            kernel1[point_q] = ene2_ptr[point_q]*(y[point_q_y] + 1j*y[point_q_y+1] + y[point_q_y+2] + 1j*y[point_q_y+3])
            kernel2[point_q] = ene1_ptr[point_q]*(y[point_q_y] + 1j*y[point_q_y+1] - y[point_q_y+2] - 1j*y[point_q_y+3])

            kernel0_a[point_q] = ene2_ptr[point_q] * (y[point_q_y] + 1j*y[point_q_y+1] + y[point_q_y+2] + 1j*y[point_q_y+3])
            kernel1_a[point_q] = ene2_ptr[point_q] * px * (y[point_q_y] + 1j*y[point_q_y+1] + y[point_q_y+2] + 1j*y[point_q_y+3])
            kernel2_a[point_q] = ene2_ptr[point_q] * py * (y[point_q_y] + 1j*y[point_q_y+1] + y[point_q_y+2] + 1j*y[point_q_y+3])

            aiy+=1
        aix+=1

    cdef double complex* temp = <double complex*> malloc(ntot*ntot*sizeof(double complex))

    ft.convolved(pdata, result1, temp, potential_c, kernel1, n2) 
    ft.convolved(pdata, result2, temp, potential_c, kernel2, n2)
    
    ft.convolved(pdata, result1_a, temp, potential_ca1, kernel1_a, n2)
    ft.convolved(pdata, result2_a, temp, potential_ca2, kernel2_a, n2)
    ft.convolved(pdata, result0_a, temp, potential_ca0, kernel0_a, n2)

    ix = 0
    while ix < ntot:
        iy = 0
        while iy < ntot:
            nx = (ix - n2)
            ny = (iy - n2)
            px = M_PI/l*nx
            py = M_PI/l*ny

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

            point_ene = (ix)*ntot + (iy)
            kgfieldc = (y[point + 0*2 + 0] + 1j*y[point + 0*2 + 1] + y[point + 1*2 + 0] + 1j*y[point + 0*2 + 1])

            conv_a0 = ene2_ptr[point_ene]*result0_a[ix*ntot+iy]
            conv_a1 = -2*ene2_ptr[point_ene]*result1_a[ix*ntot+iy]
            conv_a2 = -2*ene2_ptr[point_ene]*result2_a[ix*ntot+iy]

            tempa0c = (-1j)*(conv_a0+conv_a1+conv_a2)
            tempa1c = (-1j)*(conv_a0+conv_a1+conv_a2)

            dy[point + 0*2 + 0] += tempa0c.real
            dy[point + 0*2 + 1] += tempa0c.imag
            dy[point + 1*2 + 0] += tempa1c.real
            dy[point + 1*2 + 1] += tempa1c.imag

            iy += 1
        ix += 1
    
    free(kernel1)
    free(kernel2)
    free(result1)
    free(result2)
    free(kernel0_a)
    free(kernel1_a)
    free(kernel2_a)
    free(result0_a)
    free(result1_a)
    free(result2_a)
    free(potential_c)
    free(potential_ca1)
    free(potential_ca2)
    free(potential_ca0)
    free(temp)


def solver_t(tuple t_span, double[:] y0, double[:] coef, double[:] timesteps, 
           double[:] field, double[:] field_a1, double[:] field_a2, double[:] field_a0):

    cdef DiffeqFuncType diffeq = cython_diffeq_t
        
    cdef double* y0_ptr       = &y0[0]
    cdef unsigned int num_y   = len(y0)
    cdef int n_tsteps         = len(timesteps)
    cdef int n2               = <int>coef[0]
    cdef int n_t              = <int>coef[1]
    cdef int is_field         = <int>coef[4]
    cdef int is_field_a1      = <int>coef[5]
    cdef int is_field_a2      = <int>coef[6]
    cdef int is_field_a0      = <int>coef[7]
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

    cdef double complex* potential_c_interp = <double complex*>malloc(4*n_t*ntot*ntot*sizeof(double complex))
    cdef double complex* potential_ca1_interp = <double complex*>malloc(4*n_t*ntot*ntot*sizeof(double complex))
    cdef double complex* potential_ca2_interp = <double complex*>malloc(4*n_t*ntot*ntot*sizeof(double complex))
    cdef double complex* potential_ca0_interp = <double complex*>malloc(4*n_t*ntot*ntot*sizeof(double complex))

    cdef ft.fft_data data
    ft.setup(&data, ntot, ntot)

    cdef parameters_t args
    cdef parameters_t* args_param_ptr = &args
    args.n2 = n2
    args.n_t = n_t
    args.pdata = &data
    args.m = coef[2]
    args.l = coef[3]
    args.is_field    = is_field
    args.is_field_a1 = is_field_a1
    args.is_field_a2 = is_field_a2
    args.is_field_a0 = is_field_a0

    cdef double* t_knots = <double*>malloc(n_t*sizeof(double))
    cdef int n_k = 0
    while n_k < n_t:
        t_knots[n_k] = t_span[0] + n_k*(t_span[1]-t_span[0])/(n_t-1)
        n_k += 1
    args.t_knots = t_knots

    cdef int ix
    cdef int iy
    cdef int it
    cdef int ip

    if is_field:
        ix = 0
        while ix<ntot:
            iy = 0
            while iy<ntot:
                it = 0
                while it<n_t:
                    ip = 0
                    while ip<4:
                        potential_c_interp[ip*n_t*ntot*ntot + it*ntot*ntot + ix*ntot +iy] = field[ip*n_t*ntot*ntot*2 + it*ntot*ntot*2 + ix*ntot*2 + iy*2] + 1j*field[ip*n_t*ntot*ntot*2 + it*ntot*ntot*2 + ix*ntot*2 + iy*2 + 1]
                        ip+=1
                    it+=1
                iy+=1
            ix+=1
    
    if is_field_a1:
        ix = 0
        while ix<ntot:
            iy = 0
            while iy<ntot:
                it = 0
                while it<n_t:
                    ip = 0
                    while ip<4:
                        potential_ca1_interp[ip*n_t*ntot*ntot + it*ntot*ntot + ix*ntot +iy] = field_a1[ip*n_t*ntot*ntot*2 + it*ntot*ntot*2 + ix*ntot*2 + iy*2] + 1j*field_a1[ip*n_t*ntot*ntot*2 + it*ntot*ntot*2 + ix*ntot*2 + iy*2 + 1]
                        ip+=1
                    it+=1
                iy+=1
            ix+=1

    if is_field_a2:
        ix = 0
        while ix<ntot:
            iy = 0
            while iy<ntot:
                it = 0
                while it<n_t:
                    ip = 0
                    while ip<4:
                        potential_ca2_interp[ip*n_t*ntot*ntot + it*ntot*ntot + ix*ntot +iy] = field_a2[ip*n_t*ntot*ntot*2 + it*ntot*ntot*2 + ix*ntot*2 + iy*2] + 1j*field_a2[ip*n_t*ntot*ntot*2 + it*ntot*ntot*2 + ix*ntot*2 + iy*2 + 1]
                        ip+=1
                    it+=1
                iy+=1
            ix+=1

    if is_field_a0:
        ix = 0
        while ix<ntot:
            iy = 0
            while iy<ntot:
                it = 0
                while it<n_t:
                    ip = 0
                    while ip<4:
                        potential_ca0_interp[ip*n_t*ntot*ntot + it*ntot*ntot + ix*ntot +iy] = field_a0[ip*n_t*ntot*ntot*2 + it*ntot*ntot*2 + ix*ntot*2 + iy*2] + 1j*field_a0[ip*n_t*ntot*ntot*2 + it*ntot*ntot*2 + ix*ntot*2 + iy*2 + 1]
                        ip+=1
                    it+=1
                iy+=1
            ix+=1

    ### Test

        # t_knots = np.linspace(0,150,10)
        # tinlin = np.linspace(0,150,70)
        # nar = np.zeros(len(tinlin)).astype(complex)

        # for i_t in range(len(tinlin)):
        #     for i_k in range(len(t_knots)):
        #         if t_knots[i_k] <= tinlin[i_t] < t_knots[i_k +1] or i_k == n_t-2:
        #             nar[i_t] = sum([potential_ca1_interp[p*n_t*ntot*ntot+i_k*ntot*ntot+n2*ntot+n2]*(tinlin[i_t]-t_knots[i_k])**(3-p) for p in range(4)]) + 1j

        #             print(i_t,i_k)
        #             break

        # print(nar)
        # plt.plot(tinlin,nar.real)
        # plt.plot(tinlin,nar.imag)
        # plt.show()

    ###

    args.field_interp    = potential_c_interp
    args.field_a1_interp = potential_ca1_interp
    args.field_a2_interp = potential_ca2_interp
    args.field_a0_interp = potential_ca0_interp

    args.ene1_ptr = create_ene1(n2,coef[2],coef[3])
    args.ene2_ptr = create_ene2(n2,coef[2],coef[3])

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