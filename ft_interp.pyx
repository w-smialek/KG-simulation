from cpython cimport array
import array
from libc.math cimport sqrt, M_PI, cos, sin
import numpy as np

cdef double psixy(double* px_ptr, double* py_ptr, double* phi_ptr, int n2fac, double xx, double yy)noexcept nogil:
    cdef double[2] psi_xy
    '''Convert feshbach - villard representation 2d complex field into a 1D python array,
    with index = nx * (Ntot x 2 x 2) + ny * (2 x 2) + l * 2 + c '''

    cdef int a = -n2fac
    cdef int b = -n2fac
    while a < n2fac:
        b = -n2fac
        while b < n2fac:
            psi_xy[0] += phi_ptr[d,a,b] * cos(px_ptr[a,b]*xx + py_ptr[a,b]*yy)
            psi_xy[1] += phi_ptr[d,a,b] * sin(px_ptr[a,b]*xx + py_ptr[a,b]*yy)
            psi_xy[0] += phi_ptr[d,a,b] * cos(px_ptr[a,b]*xx + py_ptr[a,b]*yy)
            psi_xy[1] += phi_ptr[d,a,b] * sin(px_ptr[a,b]*xx + py_ptr[a,b]*yy)

    cdef int ix = 0
    cdef int iy = 0
    cdef int l  = 0
    cdef double xx = 0.
    # nx * (Ntot x 2 x 2) + ny * (2 x 2) + l * 2 + c
    while ix < n2fac:
        iy = 0
        while iy < n2fac:

            psi_xy[0] += phi_ptr[ix*n2fac*2*2 + iy*2*2 + 0*2 + 0] * cos(px_ptr[a,b]*xx + py_ptr[a,b]*yy)
            psi_xy[1] += phi_ptr[d,a,b] * sin(px_ptr[a,b]*xx + py_ptr[a,b]*yy)
            psi_xy[0] += phi_ptr[d,a,b] * cos(px_ptr[a,b]*xx + py_ptr[a,b]*yy)
            psi_xy[1] += phi_ptr[d,a,b] * sin(px_ptr[a,b]*xx + py_ptr[a,b]*yy)

            dy[ix*ntot*2*2 + iy*2*2 + 0*2 + 0] =  ene(px,py,mass) *   1.  * y[ix*ntot*2*2 + iy*2*2 + 0*2 + 1]
            dy[ix*ntot*2*2 + iy*2*2 + 0*2 + 1] = -ene(px,py,mass) *   1.  * y[ix*ntot*2*2 + iy*2*2 + 0*2 + 0]
            dy[ix*ntot*2*2 + iy*2*2 + 1*2 + 0] =  ene(px,py,mass) * (-1.) * y[ix*ntot*2*2 + iy*2*2 + 1*2 + 1]
            dy[ix*ntot*2*2 + iy*2*2 + 1*2 + 1] = -ene(px,py,mass) * (-1.) * y[ix*ntot*2*2 + iy*2*2 + 1*2 + 0]

            iy += 1
        ix += 1


    return psi_xy



def phi_to_psi_interp_cy(phi_interp_ar, px_interp_ar, py_interp_ar, factor, n2, ll):
    ntot = 2*n2

    for iy in range(-n2*factor,n2*factor):      # ROWS ARE Y,       FROM ROW     0 -> iy = -N2 -> Y = -L
        for jx in range(-n2*factor,n2*factor):  # COLLUNMS ARE X,   FROM COLLUMN 0 -> jx = -N2 -> X = -L
            for d in range(2):                  # DEPTH IS l,       d = 0 -> l = 1;  d = 1 -> q = -1
                row = n2*factor + iy 
                col = n2*factor + jx
                xx = jx/factor/n2*ll
                yy = iy/factor/n2*ll

                psi_xy = 0
                for a in range(-n2*factor,n2*factor):
                    for b in range(-n2*factor,n2*factor):
                        psi_xy += phi[d,a,b] * (cos(px_interp[a,b]*xx + py_interp[a,b]*yy) + 1j* sin(px_interp[a,b]*xx + py_interp[a,b]*yy))
                psi[d,row,col] = psi_xy
        print(iy)
    varphi, idtvarphi = 1/sqrt(2)*(psi[0,:,:]+psi[1,:,:]), 1/sqrt(2)*(psi[0,:,:]-psi[1,:,:])
    return varphi, idtvarphi
