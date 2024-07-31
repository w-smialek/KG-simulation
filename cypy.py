import numpy as np
import matplotlib.pyplot as plt
from fastccolor.colorize import colorize
from array import array
from cy_solver import solver
from PIL import Image, ImageOps
from time import time

###
###     THESE NEED TO BE CHANGED IN CY_SOLVER TOO!
###

m = 1               # mass in multiples of m_e

L = 40              # length of 1-sphere in x and y in multiples of hbar/(m_e c)
N2 = 40             # max positive/negative mode in px and py   # max 41
Ntot = 2*N2+1       # Total number of modes in one dimension

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

def fv_to_field(phi_bar):
    phi = np.zeros(phi_bar.shape).astype(complex)
    psi = np.zeros(phi_bar.shape).astype(complex)
    for nx in range(-N2,N2+1):
        for ny in range(-N2,N2+1):
            phi[:,ny+N2,nx+N2] = t_space_rot(nx,ny)@phi_bar[:,ny+N2,nx+N2]
    for l in range(2):
        phi_shifted = np.roll(phi[l,:,:],(N2+1,N2+1),(0,1))
        psi[l,:,:] = Ntot**2*np.fft.ifft2(phi_shifted)
        psi[l,:,:] = np.fft.fftshift(psi[l,:,:])
    return 1/np.sqrt(2)*(psi[0,:,:]+psi[1,:,:])

def flatten_for_cy(a):
    '''Convert feshbach - villard representation 2d complex field into a 1D python array,
    with index = nx * (Ntot x 2 x 2) + ny * (2 x 2) + l * 2 + c '''

    a_re = a.real.astype(float)
    a_im = a.imag.astype(float)

    a_out = np.zeros((2*2*Ntot*Ntot))

    for nx in range(Ntot):
        for ny in range(Ntot):
            for l in range(2):
                a_out[nx*Ntot*2*2 + ny*2*2 + l*2 +0] = a_re[l,nx,ny]
                a_out[nx*Ntot*2*2 + ny*2*2 + l*2 +1] = a_im[l,nx,ny]

    # a_out = a.reshape((-1,),order='F')
    # a_out = a_out.view(float).reshape((-1,),order='F')
    a_out = array('d',a_out)
    return a_out

### Flatten for cy and inverse test:
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

def cy_to_numpy(a):
    '''inverse of flatten_for_cy'''
    # a_out_re = np.reshape(a[0::2],(2,Ntot,Ntot),order="F").astype(complex)
    # a_out_im = np.reshape(a[1::2],(2,Ntot,Ntot),order="F").astype(complex)

    a_out_re = np.zeros((2,Ntot,Ntot)).astype(complex)
    a_out_im = np.zeros((2,Ntot,Ntot)).astype(complex)

    for nx in range(Ntot):
        for ny in range(Ntot):
            for l in range(2):
                a_out_re[l,nx,ny] = a[nx*Ntot*2*2 + ny*2*2 + l*2 +0]
                a_out_im[l,nx,ny] = a[nx*Ntot*2*2 + ny*2*2 + l*2 +1]

    return a_out_re + 1j*a_out_im

###
###     Test single mode solution
###

# phi_bar = np.zeros((2,Ntot,Ntot)).astype(complex)
# nx0 = 7
# ny0 = -2
# px0 = 2*np.pi/L*nx0
# py0 = 2*np.pi/L*ny0
# A = [1+0.5j,-2-1j]
# phi_bar[0,N2+ny0,N2+nx0] = A[0]
# phi_bar[1,N2+ny0,N2+nx0] = A[1]

# l0 = 0

# ll = 2*(-l0 + 1/2)

# pb_array = flatten_for_cy(phi_bar)
# coefs = array('d',[0.02, 3.1])
# t_span = (0., 1.0)

# result = solver(t_span, pb_array, coefs)

# print("Was Integration was successful?", result.success)
# print(result.message)
# print("Size of solution: ", result.size)

# mode_solution_r = []
# mode_solution_i = []
# for i,t in enumerate(result.t):
#     mode_solution_r.append(cy_to_numpy(result.y[:,i])[l0,N2+ny0,N2+nx0].real)
#     mode_solution_i.append(cy_to_numpy(result.y[:,i])[l0,N2+ny0,N2+nx0].imag)

# plt.plot(result.t,mode_solution_r)
# plt.plot(result.t,mode_solution_i)
# plt.show()

# mode_solution_r = []
# mode_solution_i = []
# for i,t in enumerate(result.t):
#     expp = A[l0]*np.exp(-ll*1j*t*ene(px0,py0))
#     mode_solution_r.append(expp.real)
#     mode_solution_i.append(expp.imag)

# print(ene(px0,py0))

# plt.plot(result.t,mode_solution_r)
# plt.plot(result.t,mode_solution_i)
# plt.show()

###
###     Test entire field and save gif
###

p_extent = 2*np.pi/L*N2

space_l, space_ny, space_nx = np.meshgrid(range(2), range(-N2,N2+1), range(-N2,N2+1), indexing='ij')
space_l, space_py, space_px = np.meshgrid(range(2), np.linspace(-p_extent,p_extent,Ntot), np.linspace(-p_extent,p_extent,Ntot), indexing='ij')

# b = 0.1
# kx = 0.3
# ky = 0.5
# phi_bar = (space_l+1j)*np.exp(-b*((space_nx-nx0)**2 + (space_ny-ny0)**2) + 1j*10*b*(space_nx*kx+space_ny*ky))
phi_bar = np.exp(-1j*(-20*np.pi/Ntot*space_nx) -1j*(20*np.pi/Ntot*space_ny))*(1-space_l) + np.exp(-1j*(-20*np.pi/Ntot*space_nx) -1j*(20*np.pi/Ntot*space_ny))*space_l

pb_array = flatten_for_cy(phi_bar)
coefs = array('d',[0.02, 3.1])
t_span = (0., 50.0)

result = solver(t_span, pb_array, coefs)
print("Was Integration was successful?", result.success)
print(result.message)
print("Size of solution: ", result.size)

# plt.imshow(colorize(cy_to_numpy(result.y[:,t])[0,...]), interpolation='none',extent=(-L/2,L/2,-L/2,L/2),origin='lower')
# plt.savefig('./ims/fig%i.png'%i,dpi=70)

step = result.size//300
# step = 1

t0 = time()

images = []
for i,t in enumerate(range(0,result.size,step)):
    datac = colorize(cy_to_numpy(result.y[:,t])[0,...], stretch = 3)
    img = Image.fromarray((datac[:, :, :3] * 255).astype(np.uint8))
    img = ImageOps.flip(img)
    img.save('./ims/fig%i.png'%i)
    images.append(img)
images[0].save("anim.gif", save_all = True, append_images=images[1:], duration = 100, loop=0)

del images

images = []
for i,t in enumerate(range(0,result.size,step)):
    datac = colorize(fv_to_field(cy_to_numpy(result.y[:,t])), stretch = 3)
    img = Image.fromarray((datac[:, :, :3] * 255).astype(np.uint8))
    img = ImageOps.flip(img)
    img.save('./ims/afig%i.png'%i)
    images.append(img)
images[0].save("anim2.gif", save_all = True, append_images=images[1:], duration = 100, loop=0)

te = time()
print("time: %f"%(te-t0))

# filenames = []

# for i,t in enumerate(range(0,result.size,step)):
#     plt.imsave('./ims/fig%i.png'%i,colorize(cy_to_numpy(result.y[:,t])[0,...]))
#     filenames.append('./ims/fig%i.png'%i)

# images = []
# for filename in filenames:
#     images.append(imageio.imread(filename))
# imageio.mimsave('anim.gif', images, format='GIF', duration=0.01, loop=10)

# filenames2 = []

# for i,t in enumerate(range(0,result.size,step)):
#     plt.imsave(colorize(fv_to_field(cy_to_numpy(result.y[:,t]))), './ims/f2ig%i.png'%i, dpi=250, interpolation='none',extent=(-L/2,L/2,-L/2,L/2),origin='lower')
#     plt.close()
#     filenames2.append('./ims/f2ig%i.png'%i)

# images = []
# for filename in filenames2:
#     images.append(imageio.imread(filename))
# imageio.mimsave('anim2.gif', images, format='GIF', duration=0.02, loop=0)