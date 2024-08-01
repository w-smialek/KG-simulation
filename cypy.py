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

m = 1.0               # mass in multiples of m_e

L = 10              # length of 1-sphere in x and y in multiples of hbar/(m_e c)
N2 = 25             # max positive/negative mode in px and py   # max 41
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
###     Test entire field and save gif
###

# b = 0.1
# kx = 0.3
# ky = 0.5
# phi_bar = (space_l+1j)*np.exp(-b*((space_nx-nx0)**2 + (space_ny-ny0)**2) + 1j*10*b*(space_nx*kx+space_ny*ky))
phi = np.exp(-1j*(20*np.pi/Ntot*space_nx) -1j*(20*np.pi/Ntot*space_ny))*(1-space_l) + np.exp(-1j*(20*np.pi/Ntot*space_nx) -1j*(20*np.pi/Ntot*space_ny))*space_l

phi_bar = phi_to_phibar(phi)

varphi, idtvarphi = phibar_to_varphi(phi_bar)
phi_bar_ret = varphi_to_phibar(varphi, idtvarphi)

plt.imshow(phi_bar[0,...].imag, extent=(-N2,N2,-N2,N2),origin='lower')
plt.show()
plt.imshow(phi_bar_ret[0,...].imag, extent=(-N2,N2,-N2,N2),origin='lower')
plt.show()

plt.imshow(np.sum(np.abs(phi_bar),0), extent=(-N2,N2,-N2,N2),origin='lower')
plt.show()
plt.imshow(np.sum(np.abs(phi_bar_ret),0), extent=(-N2,N2,-N2,N2),origin='lower')
plt.show()

plt.imshow(np.sum(np.abs(phi_bar - phi_bar_ret),0), extent=(-N2,N2,-N2,N2),origin='lower')
plt.show()


pb_array = flatten_for_cy(phi_bar)
coefs = array('d',[N2,L,m])

t_init = 0.
t_end = 1.0  # around 15 seconds per 1.0 on N2 = 100
n_timesteps = 50

t_span = (t_init, t_end)
timesteps = array('d',np.linspace(t_init, t_end, n_timesteps))

fps = 50

t0 = time()
result = solver(t_span, pb_array, coefs, timesteps)
te = time()
print("Mycyrk time: %f"%(te-t0))

print(result.message)
print("Size of solution: ", result.size)

t0 = time()
stretch = 2

images = []
for i in range(n_timesteps):
    datac = colorize(cy_to_numpy(result.y[:,i])[0,...], stretch)
    img = Image.fromarray((datac[:, :, :3] * 255).astype(np.uint8))
    img = ImageOps.flip(img)
    img.save('./ims/fig%i.png'%i)
    images.append(img)
    if (i%20==0):
        print(i)
images[0].save("anim.gif", save_all = True, append_images=images[1:], duration = 1/fps*1000, loop=0)

del images

images = []
for i in range(n_timesteps):
    datac = colorize(phibar_to_varphi(cy_to_numpy(result.y[:,i]))[0], stretch)
    img = Image.fromarray((datac[:, :, :3] * 255).astype(np.uint8))
    img = ImageOps.flip(img)
    img.save('./ims/afig%i.png'%i)
    images.append(img)
    if (i%20==0):
        print(i)
images[0].save("anim2.gif", save_all = True, append_images=images[1:], duration = 1/fps*1000, loop=0)

cmap1 = plt.get_cmap('binary')

images = []
for i in range(n_timesteps):
    datac = abs(phibar_to_varphi(cy_to_numpy(result.y[:,i]))[0])
    datac = datac/np.max(datac)
    datac = cmap1(datac)
    datac = np.repeat(np.repeat(datac,stretch, axis=0), stretch, axis=1)

    img = Image.fromarray((datac[:, :, :3] * 255).astype(np.uint8))
    img = ImageOps.flip(img)
    img.save('./ims/bfig%i.png'%i)
    images.append(img)
    if (i%20==0):
        print(i)
images[0].save("anim3.gif", save_all = True, append_images=images[1:], duration = 1/fps*1000, loop=0)

te = time()
print("rendering images time: %f"%(te-t0))

# t0 = time()
# stretch = 2

# fps = 50

# images = []
# for i in range(n_timesteps):
#     img = Image.open('./ims/fig%i.png'%i)
#     images.append(img)
#     if (i%20==0):
#         print(i)
# images[0].save("anim11.gif", save_all = True, append_images=images[1:], duration = 1/fps*1000, loop=0)

# del images

# images = []
# for i in range(n_timesteps):
#     img = Image.open('./ims/afig%i.png'%i)
#     images.append(img)
#     if (i%20==0):
#         print(i)
# images[0].save("anim22.gif", save_all = True, append_images=images[1:], duration = 1/fps*1000, loop=0)

# te = time()
# print("time: %f"%(te-t0))