from libc.math cimport sin, cos

cdef void cft(double* psiar, double* phiar, double* pxar, double* pyar, int factor, int n2, double ll) noexcept nogil:

    cdef int ntot = 2*n2*factor
    
    cdef int ix = 0
    cdef int iy = 0
    cdef double xx = 0.
    cdef double yy = 0.
    cdef double psi_xy_re = 0.
    cdef double psi_xy_im = 0.
    cdef int a = 0
    cdef int b = 0
    cdef int point = 0
    # nx * (Ntot x 2 x 2) + ny * (2 x 2) + l * 2 + c
    while ix < ntot:
        iy = 0
        while iy < ntot:
            xx = ll*(ix - n2*factor)/(factor*n2)
            yy = ll*(iy - n2*factor)/(factor*n2)
                
            psi_xy_re = 0.
            psi_xy_im = 0.

            point = ix*ntot*2*2 + iy*2*2

            a = 0
            while a < ntot:
                b = 0
                while b < ntot:
                    psi_xy_re += phiar[point + 0*2 + 0] * cos(pxar[point]*xx + pyar[point]*yy) - phiar[point + 0*2 + 1] * sin(pxar[point]*xx + pyar[point]*yy)
                    psi_xy_im += phiar[point + 0*2 + 0] * sin(pxar[point]*xx + pyar[point]*yy) + phiar[point + 0*2 + 1] * cos(pxar[point]*xx + pyar[point]*yy)
                    b+=1
                a+=1

            # l=0, real and imag part
            psiar[point + 0*2 + 0] = psi_xy_re
            psiar[point + 0*2 + 1] = psi_xy_im

            psi_xy_re = 0.
            psi_xy_im = 0.

            a = 0
            while a < ntot:
                b = 0
                while b < ntot:
                    psi_xy_re += phiar[point + 1*2 + 0] * cos(pxar[point+2]*xx + pyar[point+2]*yy) - phiar[point + 1*2 + 1] * sin(pxar[point+2]*xx + pyar[point+2]*yy)
                    psi_xy_im += phiar[point + 1*2 + 0] * sin(pxar[point+2]*xx + pyar[point+2]*yy) + phiar[point + 1*2 + 1] * cos(pxar[point+2]*xx + pyar[point+2]*yy)
                    b+=1
                a+=1

            # l=1, real and imag part
            psiar[point + 1*2 + 0] = psi_xy_re
            psiar[point + 1*2 + 1] = psi_xy_im

            iy += 1
        ix += 1


def ftinterp(double[:] psiar, double[:] phiar, double[:] pxar, double[:] pyar, int factor, int n2, double ll):

    cdef double* psi_ptr = &psiar[0]
    cdef double* phi_ptr = &phiar[0]
    cdef double* px_ptr  = &pxar[0]
    cdef double* py_ptr  = &pyar[0]

    cft(psi_ptr, phi_ptr, px_ptr, py_ptr, factor, n2, ll)

    return psiar