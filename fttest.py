import numpy as np
import matplotlib.pyplot as plt
from fastccolor.colorize import colorize

###
###     THESE NEED TO BE CHANGED IN CY_SOLVER TOO!
###

m = 1               # mass in multiples of m_e

L = 10              # length of 1-sphere in x and y in multiples of hbar/(m_e c)
N2 = 100             # max positive/negative mode in px and py   # max 41
Ntot = 2*N2       # Total number of modes in one dimension

print("Full size of vector: ",Ntot*Ntot*2*2)

p_extent = 2*np.pi/L*N2

space_l, space_ny, space_nx = np.meshgrid(range(2), range(-N2,N2), range(-N2,N2), indexing='ij')
space_l, space_py, space_px = np.meshgrid(range(2), np.linspace(-p_extent,p_extent,Ntot), np.linspace(-p_extent,p_extent,Ntot), indexing='ij')

###
###     Functions
###
  
def ene(px,py):
    return np.sqrt(m**2 + px**2 + py**2)

def t_space_rot(nx,ny):
    px = 2*np.pi/L*nx
    py = 2*np.pi/L*ny
    U = 1/(2*np.sqrt(m*ene(px,py)))*np.array([[m+ene(px,py),m-ene(px,py)],[m-ene(px,py),m+ene(px,py)]])
    return U

def t_space_rot_inv(nx,ny):
    px = 2*np.pi/L*nx
    py = 2*np.pi/L*ny
    U = 1/(2*np.sqrt(m*ene(px,py)))*np.array([[m+ene(px,py),ene(px,py)-m],[ene(px,py)-m,m+ene(px,py)]])
    return U

def phi_to_phibar(phi):
    phibar = np.zeros(phi.shape).astype(complex)
    for nx in range(-N2,N2):
        for ny in range(-N2,N2):
            phibar[:,ny+N2,nx+N2] = t_space_rot_inv(nx,ny)@phi[:,ny+N2,nx+N2]
    return phibar

def varphi_to_phibar(varphi, idtvarphi):
    psi = np.zeros((2,Ntot,Ntot)).astype(complex)
    psi[0,...] = 1/np.sqrt(2)*(varphi+idtvarphi)
    psi[1,...] = 1/np.sqrt(2)*(varphi-idtvarphi)

    phi = np.zeros(psi.shape).astype(complex)
    phibar = np.zeros(psi.shape).astype(complex)

    for l in range(2):
        psi[l,:,:] = np.fft.fftshift(psi[l,:,:])
        phi[l,:,:] = np.fft.fft2(psi[l,:,:])
        phi[l,:,:] = np.fft.fftshift(phi[l,:,:])

    for nx in range(-N2,N2):
        for ny in range(-N2,N2):
            phibar[:,ny+N2,nx+N2] = t_space_rot_inv(nx,ny)@phi[:,ny+N2,nx+N2]

    return phibar

def phibar_to_varphi(phi_bar):
    phi = np.zeros(phi_bar.shape).astype(complex)
    psi = np.zeros(phi_bar.shape).astype(complex)
    for nx in range(-N2,N2):
        for ny in range(-N2,N2):
            phi[:,ny+N2,nx+N2] = t_space_rot(nx,ny)@phi_bar[:,ny+N2,nx+N2]
    for l in range(2):
        phi[l,:,:] = np.roll(phi[l,:,:],(N2,N2),(0,1))
        psi[l,:,:] = np.fft.ifft2(phi[l,:,:])*np.exp(1j*np.pi*(space_nx+space_ny)[0,...])
        psi[l,:,:] = np.fft.fftshift(psi[l,:,:])
    varphi, idtvarphi = 1/np.sqrt(2)*(psi[0,:,:]+psi[1,:,:]), 1/np.sqrt(2)*(psi[0,:,:]-psi[1,:,:])
    return varphi, idtvarphi

###
###
###

phi = np.exp(-1j*(5*np.pi/Ntot*space_nx) -1j*(-5*np.pi/Ntot*space_ny))*(1-space_l) + np.exp(-1j*(-3*np.pi/Ntot*space_nx) -1j*(30*np.pi/Ntot*space_ny))*(1-space_l)

phi_bar = phi_to_phibar(phi)

varphi, idtvarphi = phibar_to_varphi(phi_bar)
phi_bar_ret = varphi_to_phibar(varphi, idtvarphi)


plt.imshow(phi_bar[0,...].imag, extent=(-N2,N2,-N2,N2),origin='lower')
plt.show()
plt.imshow(phi_bar_ret[0,...].imag, extent=(-N2,N2,-N2,N2),origin='lower')
plt.show()
plt.imshow(np.sum(np.abs(phi_bar - phi_bar_ret),0), extent=(-N2,N2,-N2,N2),origin='lower')
plt.show()
plt.imshow(abs(varphi), extent=(-N2,N2,-N2,N2),origin='lower')
plt.show()

from scipy.interpolate import RegularGridInterpolator

x, y = np.meshgrid(range(-N2,N2),range(-N2,N2))

z = phi_bar[0,...].imag

# 2D grid interpolation
interpolator_real = RegularGridInterpolator((x, y), np.real(z))
interpolator_imag = RegularGridInterpolator((x, y), np.imag(z))

def interpolator_complex(x, y):
    return interpolator_real(x, y) + 1j*interpolator_imag(x, y)

# test
new_x = np.linspace(-N2, N2, 2*Ntot)
new_y = np.linspace(-N2, N2, 2*Ntot)

ar = interpolator_complex(new_x, new_y)

plt.imshow(ar)
plt.show()