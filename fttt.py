from fastccolor.colorize import colorize
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
N2 = 100
Ntot = 2*N2
factor = 3
L = 50

space_iy, space_jx = np.meshgrid(range(-N2,N2), range(-N2,N2), indexing='ij')
spacef_iy, spacef_jx = np.meshgrid(range(-N2*factor,N2*factor), range(-N2*factor,N2*factor), indexing='ij')

space_px = np.zeros((2,Ntot,Ntot))
space_py = np.zeros((2,Ntot,Ntot))

space_x = np.zeros((2,Ntot,Ntot))
space_y = np.zeros((2,Ntot,Ntot))
space_l = np.zeros((2,Ntot,Ntot))

def p(n):
    return 2*np.pi/(2*L)*n
def x(n):
    return n/N2*L

for iy in range(-N2,N2):      # ROWS ARE Y,       FROM ROW     0 -> iy = -N2 -> Y = -L
    for jx in range(-N2,N2):  # COLLUNMS ARE X,   FROM COLLUMN 0 -> jx = -N2 -> X = -L
        for d in range(2):    # DEPTH IS l,       d = 0 -> l = 1;  d = 1 -> q = -1
            row = N2 + iy 
            col = N2 + jx
            l = 1 - 2*d

            space_px[d,row,col] = p(jx)
            space_py[d,row,col] = p(iy)

            space_x[d,row,col] = x(jx)
            space_y[d,row,col] = x(iy)
            space_l[d,row,col] = l

x0 = 25
y0 = 25

phi = (1+space_l)*np.exp(-1j*(x0*space_px + y0*space_py) - 0.9*((space_px + 1)**2 + (space_py - 1)**2))


def phi_to_psi(phi):
    psi = np.zeros(phi.shape).astype(complex)

    for d in range(2):
        psi[d,...] = Ntot**2*np.fft.ifft2(phi[d,...])*np.exp(1j*np.pi*(space_jx+space_iy))
        psi[d,...] = np.roll(psi[d,...],(N2,N2),(0,1))

    varphi, idtvarphi = 1/np.sqrt(2)*(psi[0,:,:]+psi[1,:,:]), 1/np.sqrt(2)*(psi[0,:,:]-psi[1,:,:])
    return varphi, idtvarphi

p_extent_lo = p(-N2)
p_extent_hi = p(N2-1)
p_linspace = np.linspace(p_extent_lo,p_extent_hi,Ntot)
p_linspace_interp = np.linspace(p_extent_lo,p_extent_hi,Ntot*factor)
py_interp, px_interp = np.meshgrid(p_linspace_interp, p_linspace_interp, indexing='ij')

def complex_interp_phi(phi, factor):
    phi_interp = np.zeros((2,Ntot*factor,Ntot*factor)).astype(complex)
    intplt0re = interpolate.RegularGridInterpolator([p_linspace,p_linspace], phi[0,...].real)
    intplt0im = interpolate.RegularGridInterpolator([p_linspace,p_linspace], phi[0,...].imag)
    intplt1re = interpolate.RegularGridInterpolator([p_linspace,p_linspace], phi[1,...].real)
    intplt1im = interpolate.RegularGridInterpolator([p_linspace,p_linspace], phi[1,...].imag)

    phi_interp[0,...] = intplt0re((py_interp,px_interp)) + 1j*intplt0im((py_interp,px_interp))
    phi_interp[1,...] = intplt1re((py_interp,px_interp)) + 1j*intplt1im((py_interp,px_interp))
    return phi_interp

def phi_to_psi_interp(phi, factor):
    psi = np.zeros(phi.shape).astype(complex)

    for d in range(2):
        psi[d,...] = (Ntot*factor)**2*np.fft.ifft2(phi[d,...])*np.exp(1j*np.pi*(spacef_jx+spacef_iy))
        psi[d,...] = np.roll(psi[d,...],(N2*factor,N2*factor),(0,1))

    varphi, idtvarphi = 1/np.sqrt(2)*(psi[0,:,:]+psi[1,:,:]), 1/np.sqrt(2)*(psi[0,:,:]-psi[1,:,:])
    return varphi, idtvarphi

phi_interp = complex_interp_phi(phi, factor)

varphi_0, _ = phi_to_psi(phi)
varphi_interp, _ = phi_to_psi_interp(phi_interp, factor)

plt.imshow(colorize(phi[0,...]))
plt.show()

plt.imshow(colorize(phi_interp[0,...]))
plt.show()

plt.imshow(colorize(varphi_0))
plt.show()

plt.imshow(colorize(varphi_interp))
plt.show()