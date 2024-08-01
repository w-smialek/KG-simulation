import numpy as np
import matplotlib.pyplot as plt
from fastccolor.colorize import colorize

N2 = 200
Ntot = 2*10+1

X,Y = np.meshgrid(range(-N2,N2+1),range(-N2,N2+1))

kx = 1
ky = -1
Z = np.exp(-1j*2*np.pi/Ntot*(kx*X+ky*Y))

plt.imshow(colorize(Z),origin='lower',extent=(-N2,N2,-N2,N2))
plt.show()

Zft = np.fft.ifft2(Z)
Zft = np.fft.fftshift(Zft)

plt.imshow(abs(Zft),origin='lower',extent=(-N2,N2,-N2,N2))
plt.show()

print(Zft[N2+ky,N2+kx])