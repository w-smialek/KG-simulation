import numpy as np
import matplotlib.pyplot as plt
from colorsys import hls_to_rgb
from array import array
from cy_solver import solver

###
### THESE NEED TO BE CHANGED IN CY_SOLVER TOO!

m = 1
hbar = 1
c = 1

L = 20*hbar/(m*c)   # length of 1-sphere in x and y
N2 = 41             # max positive/negative mode in px and py   # max 41
Ntot = 2*N2+1       # Total number of modes in one dimension

###
###
  
def colorize(z):
    n,m = z.shape
    c = np.zeros((n,m,3))
    c[np.isinf(z)] = (1.0, 1.0, 1.0)
    c[np.isnan(z)] = (0.5, 0.5, 0.5)

    idx = ~(np.isinf(z) + np.isnan(z))
    A = (np.angle(z[idx]) + np.pi) / (2*np.pi)
    A = (A + 0.5) % 1.0
    B = 1.0 - 1.0/(1.0+abs(z[idx])**0.3)
    c[idx] = [hls_to_rgb(a, b, 0.8) for a,b in zip(A,B)]
    return c

def ene(px,py):
    return np.sqrt(m**2 + px**2 + py**2)

def t_space_rot(nx,ny):
    px = hbar*2*np.pi/L*nx
    py = hbar*2*np.pi/L*ny
    U = 1/(2*np.sqrt(m*ene(px,py)))*np.array([[m+ene(px,py),m-ene(px,py)],[m-ene(px,py),m+ene(px,py)]])
    return U

def fv_to_field(phi_bar):
    phi = np.zeros(phi_bar.shape).astype(complex)
    psi = np.zeros(phi_bar.shape).astype(complex)
    for nx in range(-N2,N2+1):
        for ny in range(-N2,N2+1):
            phi[:,nx+N2,ny+N2] = t_space_rot(nx,ny)@phi_bar[:,nx+N2,ny+N2]
    for l in range(2):
        phi_shifted = np.roll(phi[l,:,:],(N2+1,N2+1),(0,1))
        psi[l,:,:] = Ntot**2*np.fft.ifft2(phi_shifted)
        psi[l,:,:] = np.fft.fftshift(psi[l,:,:])
    return 1/np.sqrt(2)*(psi[0,:,:]+psi[1,:,:])

def flatten_for_cy(a):
    '''Convert feshbach - villard representation 2d complex field into a 1D python array,
    with index = nx * (Ntot x 2 x 2) + ny * (2 x 2) + l * 2 + c '''

    a_out = a.reshape((-1,),order='F')
    a_out = a_out.view(float).reshape((-1,),order='F')
    a_out = array('d',a_out)
    return a_out

def cy_to_numpy(a):
    '''inverse of flatten_for_cy'''
    a_out_re = np.reshape(a[0::2],(2,Ntot,Ntot),order="F").astype(complex)
    a_out_im = np.reshape(a[1::2],(2,Ntot,Ntot),order="F").astype(complex)

    return a_out_re + 1j*a_out_im


p_extent = 2*np.pi*hbar/L*N2

space_nx, space_ny, space_l = np.meshgrid(range(-N2,N2+1), range(2), range(-N2,N2+1))
space_px, space_py, space_l = np.meshgrid(np.linspace(-p_extent,p_extent,Ntot), range(2), np.linspace(-p_extent,p_extent,Ntot))


phi_bar = np.zeros(space_nx.shape).astype(complex)
nx0 = 2
ny0 = 2
phi_bar[0,N2+ny0,N2+nx0] = 1j
nx0 = 3
ny0 = -2
phi_bar[1,N2+ny0,N2+nx0] = -1
# b = 0.1
# kx = 0.3
# ky = 0.5
# phi_bar = (space_l+1j)*np.exp(-b*((space_nx-nx0)**2 + (space_ny-ny0)**2) + 1j*10*b*(space_nx*kx+space_ny*ky))
# phi_bar = np.exp(1j*space_nx)

# phitest = flatten_for_cy(phi_bar)
# for nx in range(Ntot):
#     for ny in range(Ntot):
#         for l in range(2):
#             for c in range(2):
#                 if c == 0:
#                     print(phitest[nx*Ntot*2*2 + ny*2*2 + l*2 +c])
#                     print(phi_bar[l,ny,nx].real)
#                 if c == 1:
#                     print(phitest[nx*Ntot*2*2 + ny*2*2 + l*2 +c])
#                     print(phi_bar[l,ny,nx].imag)

pb_array = flatten_for_cy(phi_bar)
coefs = array('d',[0.02, 3.1])
t_span = (0., 3.0)

result = solver(t_span, pb_array, coefs)


print("Was Integration was successful?", result.success)
print(result.message)
print("Size of solution: ", result.size)

# plt.imshow(colorize(fv_to_field(phi_bar)), interpolation='none',extent=(-L/2,L/2,-L/2,L/2),origin='lower')
# plt.show()
# plt.imshow(colorize(fv_to_field(cy_to_numpy(result.y[:,0]))), interpolation='none',extent=(-L/2,L/2,-L/2,L/2),origin='lower')
# plt.show()

filenames = []

for t in range(0,result.size,result.size//70):
    plt.imshow(colorize(fv_to_field(cy_to_numpy(result.y[:,t]))), interpolation='none',extent=(-L/2,L/2,-L/2,L/2),origin='lower')
    plt.savefig('./ims/fig%i.png'%t,dpi=70)
    filenames.append('./ims/fig%i.png'%t)

import imageio
images = []
for filename in filenames:
    images.append(imageio.imread(filename))
imageio.mimsave('anim.gif', images, format='GIF', duration=0.01, loop=10)

# import os

# for filename in filenames:
#     os.remove(filename)