#ifndef FFT_STUFF_H
#define FFT_STUFF_H
// REALLY IMPORTANT TO INCLUDE complex.h before fftw3.h!!!
// forces it to use the standard C complex number definition
#include<complex>
#include<fftw3.h>

typedef struct fft_data {
	int N1;
	int N2;
	fftw_plan plan_forward;
	fftw_plan plan_backward;
	fftw_complex *data_in;
	fftw_complex *data_out;
} fft_data;

void setup(fft_data *data, int N1, int N2);

void finalise(fft_data *data);

void execute_transform_forward(fft_data *data);

void execute_transform_backward(fft_data *data);

void fill_in(fft_data* data, std::complex<double>* array);

void read_out(fft_data* data, std::complex<double>* array);

#endif
