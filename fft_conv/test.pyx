cimport src.fftlib as ft
cimport numpy as np
import numpy as np
import matplotlib.pyplot as plt
from time import time
from libc.stdio cimport printf

from libc.stdlib cimport malloc, free

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

cdef void convolved(double complex* result, double complex* temp, double complex* field, double complex* kernel, int n2) noexcept nogil:

    cdef int ntot = 2*n2
    cdef int aix = 0
    cdef int aiy = 0
    cdef int count = 0
    cdef ft.fft_data data
    cdef int i = 0
    ft.setup(&data,ntot,ntot)
    ft.fill_in(&data,field)
    ft.execute_transform_forward(&data)
    ft.read_out(&data,temp)
    ft.fill_in(&data,kernel)
    ft.execute_transform_forward(&data)
    ft.read_out(&data,result)
    i = 0
    while i<ntot*ntot:
        temp[i] *= 1/(ntot*ntot)*result[i]
        i+=1
    ft.fill_in(&data,temp)
    ft.execute_transform_backward(&data)
    ft.read_out(&data,temp)
    ft.finalise(&data)

    cdef double complex z

    while aix<ntot:
        aiy=0
        while aiy<ntot:
            z = convolution(field, kernel, aix, aiy, n2)

            if abs(z - temp[((aix-n2)%ntot)*ntot+((aiy-n2)%ntot)])<0.0001:
                count+=1
                printf("%i %i\n",aix,aiy)
            else:
                count += -1
            result[aix*ntot+aiy] = z

            aiy+=1
        aix+=1
    printf("%i\n",count)
    return

import numpy as np
import matplotlib.pyplot as plt

N = 40
N2 = N//2
L = 10

X,Y = np.meshgrid(np.linspace(-L/2,L/2,N),np.linspace(-L/2,L/2,N),indexing='ij')

gauss1 = np.exp(1j*(3*X + 2*Y))
gauss2 = (0.5+0.5j)*np.exp(-0.5*((X-2.5)**2 + (Y-2.5)**2))


cdef double complex* g1 = <double complex*> malloc(N*N*sizeof(double complex))
cdef double complex* g2 = <double complex*> malloc(N*N*sizeof(double complex))
i=0
j=0
while i<N:
    j=0
    while j<N:
        g1[i*N+j] = gauss1[i,j]
        g2[i*N+j] = gauss2[i,j]
        j+=1
    i+=1

cdef double complex* temp = <double complex*>malloc(N*N*sizeof(double complex))
cdef double complex* result = <double complex*>malloc(N*N*sizeof(double complex))

convolved(result, temp, g1, g2, N2)

fftwout = np.zeros((N,N)).astype(complex)
for i in range(N):
    for j in range(N):
        fftwout[j,i] = result[i*N + j]

fftwout = np.fft.fftshift(fftwout)

plt.matshow(fftwout.real)
plt.matshow(fftwout.imag)
# plt.matshow(convout.real)
# plt.matshow(convout.imag)
plt.show()


# N = 100
# N2 = N//2
# L = 10
# cdef ft.fft_data data2

# ft.setup(&data2, N, N)

# X,Y = np.meshgrid(np.linspace(-L/2,L/2,N),np.linspace(-L/2,L/2,N),indexing='ij')

# gauss1 = np.exp(1j*(3*X + 2*Y))
# gauss2 = (0.5+0.5j)*np.exp(-0.5*((X-2.5)**2 + (Y-2.5)**2))


# cdef double complex* g1 = <double complex*> malloc(N*N*sizeof(double complex))
# cdef double complex* g2 = <double complex*> malloc(N*N*sizeof(double complex))
# i=0
# j=0
# while i<N:
#     j=0
#     while j<N:
#         g1[i*N+j] = gauss1[i,j]
#         g2[i*N+j] = gauss2[i,j]
#         j+=1
#     i+=1

# t0 = time()
# cdef double complex* convolved_loop = <double complex*> malloc(N*N*sizeof(double complex))
# i=0
# j=0
# while i<N:
#     j=0
#     while j<N:
#         convolved_loop[i*N+j] = ft.convolution(&g1[0],&g2[0],j,i,N2)
#         j+=1
#     i+=1
# t1 = time()
# print("loop time: %f"%(t1-t0))

# convout = np.zeros((N,N)).astype(complex)
# for i in range(N):
#     for j in range(N):
#         convout[i,j] = convolved_loop[i*N + j]

# ###

# t0 = time()
# ft.convolution_fftw(&data2,&g1[0],&g2[0],N2)
# t1 = time()
# print("fftw time: %f"%(t1-t0))

# ft.read_out(&data2,&g1[0])

# count = 0
# for i in range(N):
#     for j in range(N):
#         if abs(convolved_loop[i*N+j] - g1[((j-N2)%N)*N+((i-N2)%N)]) < 0.0001:
#             # print(i,j)
#             # print(((j-N2)%N),((i-N2)%N))
#             # print(convolved_loop[i*N+j], data2.data_out[((j-N2)%N)*N+((i-N2)%N)])
#             count +=1
# print(count)

# fftwout = np.zeros((N,N)).astype(complex)
# for i in range(N):
#     for j in range(N):
#         fftwout[j,i] = g1[i*N + j]

# fftwout = np.fft.fftshift(fftwout)

# plt.matshow(fftwout.real)
# plt.matshow(fftwout.imag)
# plt.matshow(convout.real)
# plt.matshow(convout.imag)
# plt.show()