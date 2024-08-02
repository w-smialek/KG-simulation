from libc.math cimport M_PI, sqrt

cdef double ene(double px,double py,double m)noexcept nogil:
    return sqrt(m*m + px*px + py*py)


# def t_space_rot(nx,ny, L, m):      # phi_p = U(p) @ phi_bar_p
#     px = 2*M_PI/L*nx
#     py = 2*M_PI/L*ny
#     U = 1/(2*sqrt(m*ene(px,py)))*np.array([[m+ene(px,py),m-ene(px,py)],[m-ene(px,py),m+ene(px,py)]])
#     return U

cdef void rot(double* ar,double L,double m,int n2)noexcept nogil:

    cdef int ntot = 2*n2

    cdef double px = 0.
    cdef double py = 0.
    cdef double dbl_nx = 0.
    cdef double dbl_ny = 0.
    cdef int ix = 0
    cdef int iy = 0
    cdef double ene_val = 0.
    cdef double pre_val = 0.
    # nx * (Ntot x 2 x 2) + ny * (2 x 2) + l * 2 + c
    while ix < ntot:
        iy = 0
        while iy < ntot:
            dbl_nx = <double>(ix - n2)
            dbl_ny = <double>(iy - n2)
            px = 2*M_PI/L*dbl_nx
            py = 2*M_PI/L*dbl_ny
            ene_val = ene(px, py, m)
            pre_val = 1/(2*sqrt(m*ene_val))

            ar[ix*ntot*2*2 + iy*2*2 + 0*2 + 0] = ((m+ene_val)* ar[ix*ntot*2*2 + iy*2*2 + 0*2 + 0] + (m-ene_val)* ar[ix*ntot*2*2 + iy*2*2 + 1*2 + 0])*pre_val
            ar[ix*ntot*2*2 + iy*2*2 + 0*2 + 1] = ((m+ene_val)* ar[ix*ntot*2*2 + iy*2*2 + 0*2 + 1] + (m-ene_val)* ar[ix*ntot*2*2 + iy*2*2 + 1*2 + 1])*pre_val
            ar[ix*ntot*2*2 + iy*2*2 + 1*2 + 0] = ((m-ene_val)* ar[ix*ntot*2*2 + iy*2*2 + 0*2 + 0] + (m+ene_val)* ar[ix*ntot*2*2 + iy*2*2 + 1*2 + 0])*pre_val
            ar[ix*ntot*2*2 + iy*2*2 + 1*2 + 1] = ((m-ene_val)* ar[ix*ntot*2*2 + iy*2*2 + 0*2 + 1] + (m+ene_val)* ar[ix*ntot*2*2 + iy*2*2 + 1*2 + 1])*pre_val

            iy += 1
        ix += 1
    return

# U = 1/(2*sqrt(m*ene(px,py)))*np.array([[m+ene(px,py),m-ene(px,py)],[m-ene(px,py),m+ene(px,py)]])


def barcy(double[:] phi_bar,double L,double m,int n2):

    phi_bar_ptr = &phi_bar[0]
    rot(phi_bar_ptr, L, m, n2)

    return phi_bar