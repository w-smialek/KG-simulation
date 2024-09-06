cimport cython

cdef extern from 'hsl.h' nogil:

    struct hsl:
        double h
        double s
        double l

    struct rgb:
        double r
        double g
        double b

    # METHOD WITH STRUCT
    hsl struct_rgb_to_hsl(double r, double g, double b)nogil;
    rgb struct_hsl_to_rgb(double h, double s, double l)nogil;

ctypedef hsl HSL_
ctypedef rgb RGB_

# from HSL cimport hsl_to_rgb
from libc.math cimport atan2, M_PI, fmod, powf, sqrt

# def colorize0(z, stretch = 1):
#     if stretch != 1:
#         z = np.repeat(np.repeat(z,stretch, axis=0), stretch, axis=1)
#     n,m = z.shape    

#     c = np.zeros((n,m,3))
#     c[np.isinf(z)] = (1.0, 1.0, 1.0)
#     c[np.isnan(z)] = (0.5, 0.5, 0.5)

#     idx = ~(np.isinf(z) + np.isnan(z))
#     A = (np.angle(z[idx]) + np.pi) / (2*np.pi)
#     A = (A + 0.5) % 1.0
#     B = 1.0 - 1.0/(1.0+abs(z[idx])**0.3)
#     c[idx] = [hsl_to_rgb(a, 0.8, b) for a,b in zip(A,B)]
#     return c

cdef void cycol(double* z, double* c, int n, int m)noexcept nogil:

    cdef int nx = 0
    cdef int ny = 0
    cdef double zim = 0.
    cdef double zre = 0.
    cdef double angle = 0.
    cdef double b = 0.
    cdef double abs_z = 0.
    cdef RGB_ rgb_

    while nx < n:
        ny = 0
        while ny < m:
            zre = z[ny*m*2 + nx*2 + 0]
            zim = z[ny*m*2 + nx*2 + 1]
            angle = (atan2(zim,zre)/(2*M_PI) + 0.5)
            if angle != angle:
                c[ny*m*3 + nx*3 + 0] = 0.5
                c[ny*m*3 + nx*3 + 1] = 0.5
                c[ny*m*3 + nx*3 + 2] = 0.5
            else:
                angle = fmod(angle + 0.5, 1.0)
                abs_z = sqrt(zim*zim + zre*zre)
                b = 1.0 - 1.0/(1.0+powf(abs_z,0.3))
                if b != b:
                    c[ny*m*3 + nx*3 + 0] = 0.5
                    c[ny*m*3 + nx*3 + 1] = 0.5
                    c[ny*m*3 + nx*3 + 2] = 0.5
                else:
                    rgb_ = struct_hsl_to_rgb(angle,0.8,b)
                    c[ny*m*3 + nx*3 + 0] = <double>rgb_.r
                    c[ny*m*3 + nx*3 + 1] = <double>rgb_.g
                    c[ny*m*3 + nx*3 + 2] = <double>rgb_.b
                    # c[ny*m*3 + nx*3 + 0] = hsl_to_rgb(angle,0.8,b)[0]
                    # c[ny*m*3 + nx*3 + 1] = hsl_to_rgb(angle,0.8,b)[1]
                    # c[ny*m*3 + nx*3 + 2] = hsl_to_rgb(angle,0.8,b)[2]
            ny+=1
        nx+=1
    return

def colorize(double[:] z,double[:] c, int n, int m):

    z_ptr = &z[0]
    c_ptr = &c[0]
    cycol(z_ptr, c_ptr, n, m)

    return c