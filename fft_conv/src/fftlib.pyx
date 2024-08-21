cimport libc.complex
from libc.stdlib cimport malloc, free
from libc.stdio cimport printf

cdef double complex convolution(double complex* field, double complex* kernel, int ix, int iy, int n2) noexcept nogil:
    cdef int ntot = n2*2
    cdef double complex conv_val = 0.+0.j
    cdef int aix = 0
    cdef int aiy = 0
    cdef int point_q = 0
    cdef int point_sh = 0
    while aix < ntot:
        aiy = 0
        while aiy < ntot:
            point_q = (aix)*ntot + (aiy)
            point_sh = ((ix-(aix-n2))%ntot)*ntot + (iy-(aiy-n2))%ntot

            conv_val += field[point_sh]*kernel[point_q]

            aiy+=1
        aix+=1
    return conv_val

cdef void convolution_fftw(fft_data* data, double complex* field, double complex* kernel, int n2) noexcept nogil:
    cdef int ntot = n2*2
    cdef double complex* temp = <double complex*> malloc(ntot*ntot*sizeof(double complex))
    cdef double complex* temp2 = <double complex*> malloc(ntot*ntot*sizeof(double complex))
    cdef int i = 0
    fill_in(data,field)
    execute_transform_forward(data)
    read_out(data,temp)
    fill_in(data,kernel)
    execute_transform_forward(data)
    read_out(data,temp2)
    i = 0
    while i<ntot*ntot:
        temp[i] *= 1/(ntot*ntot)*temp2[i]
        i+=1
    fill_in(data,temp)
    execute_transform_backward(data)
    free(temp)
    free(temp2)
    return

cdef void convolved(fft_data* data, double complex* result, double complex* temp, double complex* field, double complex* kernel, int n2) noexcept nogil:

    cdef int ntot = 2*n2
    cdef int aix = 0
    cdef int aiy = 0
    cdef int count = 0
    # cdef fft_data data
    cdef int i = 0
    # setup(&data,ntot,ntot)
    fill_in(data,field)
    execute_transform_forward(data)
    read_out(data,temp)
    fill_in(data,kernel)
    execute_transform_forward(data)
    read_out(data,result)
    i = 0
    while i<ntot*ntot:
        temp[i] *= 1/(ntot*ntot)*result[i]
        i+=1
    fill_in(data,temp)
    execute_transform_backward(data)
    read_out(data,temp)
    # finalise(&data)

    while aix<ntot:
        aiy=0
        while aiy<ntot:

            result[aix*ntot+aiy] = temp[((aix-n2)%ntot)*ntot+((aiy-n2)%ntot)]

            aiy+=1
        aix+=1
    return
