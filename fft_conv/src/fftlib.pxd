cimport libc.complex

cdef extern from "fftw3.h" nogil:
    ctypedef double[2] fftw_complex

cdef extern from "fft_stuff.h" nogil:
    struct fft_data:
        complex *data_in
        complex *data_out
    void setup(fft_data *data, int N1, int N2)
    void finalise(fft_data *data)
    void execute_transform_forward(fft_data *data)
    void execute_transform_backward(fft_data *data)
    void fill_in(fft_data* data, double complex* array)
    void read_out(fft_data* data, double complex* array)

cdef double complex convolution(double complex* field, double complex* kernel, int ix, int iy, int n2) noexcept nogil

cdef void convolution_fftw(fft_data* data, double complex* field, double complex* kernel, int n2) noexcept nogil

cdef void convolved(fft_data* data, double complex* result, double complex* temp, double complex* field, double complex* kernel, int n2) noexcept nogil