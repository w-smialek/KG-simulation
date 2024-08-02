cdef extern from 'hsl.c' nogil:

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